# -*- coding: utf-8 -*-
import numpy as np
import random
from hashlib import sha1
from scipy.spatial.distance import hamming


class MinHash(object):
    """
    MinHash (Broder 1997)
    """
    def __init__(self, number_hash_functions):
        """
        :param number_hash_functions: Int >= 1
        """
        self._mersenne_prime = (1 << 89) - 1  # (x << n) is x shifted left by n bit
        self._max_hash = (1 << 62) - 1  # BARNES: Changed from 64 --> 62
        random.seed(427)
        self._a, self._b = np.array([(random.randint(1, self._mersenne_prime), random.randint(0, self._mersenne_prime)) for _ in xrange(number_hash_functions)]).T
        self._number_hash_functions = number_hash_functions
        self.signatures = dict()

    def hash_corpus(self, file_name, delimiter=' ', headers=0, doc_id_0=0):
        """
        Apply MinHash to a raw text file, and documents to dataset
        :param file_name: String, path to file name
        :param delimiter: String to split tokens by
        :param headers: Number of header lines in file
        :param doc_id_0: Document id to assign to first document in file
        """
        doc_id = doc_id_0 - headers
        with open(file_name) as ins:
            for line in ins:
                print 'Adding document ' + str(doc_id) + ' to corpus'
                if doc_id >= doc_id_0:
                    tokens = frozenset(line.rstrip('\n').split(delimiter))
                    self.signatures[doc_id] = self._hash_document(tokens)
                doc_id += 1

    def _hash_document(self, document):
        """
        MinHash signature of a single document, does not add to dataset
        :param document: Set of tokens
        :return signature: numpy vector of MinHash signature
        """
        signature = np.empty(self._number_hash_functions)
        signature.fill(self._max_hash)
        for token in document:
            signature = np.minimum(self._hash_token(token), signature)
        return signature

    def _hash_token(self, token):
        """
        Apply all hash functions to a single token
        :param token: String
        :return values:
        """
        hv = int(sha1(token).hexdigest(), 16) % (10 ** 12)
        # Do Carter and Wegman like hashing.
        values = np.bitwise_and((self._a * hv + self._b) % self._mersenne_prime, self._max_hash)
        return values

    def jaccard(self, id1, id2):
        """
        Approximate Jaccard coefficient using minhash
        :param id1: Doc ID (key)
        :param id2: Doc ID (key)
        :return j: Approximate Jaccard coefficient
        """
        j = 1 - hamming(self.signatures[id1], self.signatures[id2])
        return j


class Banding(object):
    """
    Banding the MinHash signatures for quickly finding neighbors
    """
    def __init__(self, number_hash_functions, threshold):
        """
        :param number_hash_functions: Integer, number of hash functions
        :param threshold: Jaccard threshold in [0, 1]
        """
        self._threshold = threshold
        bandwidth = self._calculate_bandwidth(number_hash_functions, self._threshold)
        self._number_bands_per_doc = number_hash_functions / bandwidth
        self.band_to_docs = dict()
        self.doc_to_bands = dict()
        print 'Initialized bands with ' + str(self._number_bands_per_doc) + ' bands per document.'

    @property
    def number_bands(self):
        return len(self.band_to_docs)

    @property
    def number_docs_in_bands(self):
        return len(self.doc_to_bands)*self._number_bands_per_doc

    def get_threshold(self):
        """
        Returns the threshold bands were created at
        :return threshold:
        """
        return self._threshold

    def add_signatures(self, signatures):
        """
        Add multiple signatures to the banding
        :param signatures: Dictionary of [doc id, signature]
        """
        for counter, (doc_id, signature) in enumerate(signatures.iteritems()):
            print 'Banding document ' + str(counter)
            self._add_signature(doc_id, signature)
        print 'Added ' + str(len(signatures)) + ' documents to the banding. Total of ' + str(self.number_bands) + ' bands with ' + str(self.number_docs_in_bands) + ' stored doc ids (including repeated elements in different bands.'

    def _add_signature(self, doc_id, signature):
        """
        Add a single document to the banding
        :param doc_id: Document ID
        :param signature: numpy vector of a single document's minhash signature
        """
        if doc_id not in self.doc_to_bands:
            bands = set()
            for i, raw_band in enumerate(np.array_split(signature, self._number_bands_per_doc)):
                band = sha1("ab" + str(raw_band) + "ba"+str(i)).digest()
                bands.add(band)
                if band in self.band_to_docs:
                    self.band_to_docs[band].add(doc_id)
                else:
                    self.band_to_docs[band] = {doc_id}
            self.doc_to_bands[doc_id] = bands
        else:
            raise KeyError('Attempted to add same document multiple times')

    def band_to_docs(self, band_key):
        """
        :param band_key: String
        :return doc_ids: Set of document ids
        """
        return self.band_to_docs[band_key]

    def doc_to_bands(self, doc_key):
        """
        :param doc_key: Document ID
        :return bands: Set of strings, bands this document belongs to
        """
        return self.doc_to_bands[doc_key]

    @staticmethod
    def _calculate_bandwidth(number_hash_functions, threshold):
        """
        Calculates bandwidth
        b #bands
        r #rows per band
        :param n: = b * r  # elements in signature (number of hash functions)
        :param threshold: Jaccard threshold, tr = (1/b) ** (1/r)
        :return best: Integer, bandwidth
        """
        best = number_hash_functions, 1
        minerr = float("inf")
        for r in xrange(1, number_hash_functions + 1):
            try:
                b = 1. / (threshold ** r)
            except:
                return best
            err = abs(number_hash_functions - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best