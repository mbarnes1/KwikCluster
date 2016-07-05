# -*- coding: utf-8 -*-
import numpy as np
import random
from hashlib import sha1
from scipy.spatial.distance import hamming
import multiprocessing
from itertools import izip
from functools import partial
import copy_reg
import types
from sys import maxint
import gzip
import json
__author__ = 'Benedikt Boecking and Matt Barnes'


class Worker(multiprocessing.Process):
    """
    This is a multiprocessing worker, which when created starts another Python instance.
    After initialization, work begins with .start()
    When finished (determined when sentinel object - None - is queue is processed), clean up with .join()
    """
    def __init__(self, minhash, jobqueue, resultsqueue):
        """
        :param minhash: MinHash object
        :param jobqueue: Multiprocessing.Queue() of records to hash
        :param resultsqueue: Multiprocessing.Queue() of tuples.
                             tuple[0] = Doc ID
                             tuple[1] = Doc MinHash signature
        """
        super(Worker, self).__init__()
        self.job_queue = jobqueue
        self.results_queue = resultsqueue
        self.minhash = minhash

    def run(self):
        print 'Worker started'
        for job in iter(self.job_queue.get, None):
            doc_id = job[0]
            tokens = job[1]
            signature = self.minhash.hash_document(tokens)
            self.results_queue.put((doc_id, signature))
        print 'Worker exiting'


class MinHash(object):
    """
    MinHash (Broder 1997)
    """
    def __init__(self, number_hash_functions):
        """
        :param number_hash_functions: Int >= 1
        """
        self._mersenne_prime = (1 << 89) - 1  # (x << n) is x shifted left by n bit
        self._max_hash = maxint  # (1 << 64) - 1  # BARNES: Changed from 64 --> 62
        random.seed(427)
        self._a, self._b = np.array([(random.randint(1, self._mersenne_prime), random.randint(0, self._mersenne_prime)) for _ in xrange(number_hash_functions)]).T
        self._number_hash_functions = number_hash_functions
        self.signatures = dict()
        self.line_to_index = dict()

    def hash_corpus_list(self, documents, doc_id_0=0, number_threads=1, delimiter=' '):
        """
        Apply MinHash to pre-loaded document, add documents to dataset
        :param documents: List of record texts
        :param doc_id_0: Document id to assign to first document in file
        :param number_threads: Number of threads to hash documents with
        :param delimiter: String to split tokens by
        """
        jobs = []
        job_ids = []
        for doc_id, text in enumerate(documents):
            tokens = frozenset(text.rstrip('\n').split(delimiter))
            jobs.append(tokens)
            job_ids.append(doc_id + doc_id_0)
        p = multiprocessing.Pool(number_threads)
        chunk_size = int(float(len(jobs)) / number_threads)
        signatures = p.map(self.hash_document, jobs, chunk_size)
        self.signatures = {doc_id: signature for doc_id, signature in izip(job_ids, signatures)}

    def hash_corpus(self, file_name, delimiter=' ', headers=0, doc_id_0=0, number_threads=1, max_lines=np.Inf, input_gzip=False, input_json=False):
        """
        Apply MinHash to a raw text file, add documents to dataset
        :param file_name: String, path to file name
        :param delimiter: String to split tokens by
        :param headers: Number of header lines in file
        :param doc_id_0: Document id to assign to first document in file
        :param number_threads: Number of threads to hash documents with
        :param max_lines: Maximum number of lines to read in from file
        :param input_gzip: Boolean, whether input is a gzip file
        :param input_json: Boolean, whether input is a json file
        """
        doc_line = doc_id_0 - headers
        if input_gzip:
            ins = gzip.GzipFile(file_name, 'rb')
        else:
            ins = open(file_name, 'rb')
        if number_threads > 1:
            job_queue = multiprocessing.Queue(10000)
            results_queue = multiprocessing.Queue(20000)
            worker_pool = list()
            for _ in range(number_threads):
                w = Worker(self, job_queue, results_queue)
                worker_pool.append(w)
                w.start()
            number_jobs = 0
            number_finished_jobs = 0
            for line in ins:
                if doc_line >= doc_id_0:
                    if doc_line % 1000 == 0:
                        print 'Reading document ' + str(doc_line) + '. Simultaneously hashing in parallel.'
                    if input_json:
                        json_object = json.loads(line)
                        tokens = json_object["_source"]['extracted_text']
                        doc_index = json_object["_id"]
                    else:
                        tokens = frozenset(line.rstrip('\n').split(delimiter))
                        doc_index = doc_line
                    self.line_to_index[doc_line] = doc_index
                    job_queue.put((doc_line, tokens))
                    number_jobs += 1
                    if number_jobs >= max_lines:
                        break
                    if number_jobs > number_finished_jobs + 5000:
                        result = results_queue.get()
                        self.signatures[result[0]] = result[1]
                        number_finished_jobs += 1
                        if number_finished_jobs % 1000 == 0:
                            print 'Emptying Minhash results queue: ' + str(number_finished_jobs) + ' emptied results'
                doc_line += 1
            for _ in worker_pool:
                job_queue.put(None)  # Sentinel objects to allow clean shutdown: 1 per worker.
            while number_finished_jobs < number_jobs:
                result = results_queue.get()
                doc_line = result[0]
                signature = result[1]
                self.signatures[doc_line] = signature
                number_finished_jobs += 1
                if number_finished_jobs % 1000 == 0:
                    print 'Emptying Minhash results queue: ' + str(number_finished_jobs) + ' of ' + str(number_jobs)
            print 'Joining workers'
            for worker in worker_pool:
                worker.join()
        else:
            for line in ins:
                if doc_line % 1000 == 0:
                    print 'Adding document ' + str(doc_line) + ' to corpus'
                if doc_line >= doc_id_0:
                    if input_json:
                        json_object = json.loads(line)
                        print json_object
                        tokens = json_object["_source"]['extracted_text']
                        doc_index = json_object["_id"]
                    else:
                        tokens = frozenset(line.rstrip('\n').split(delimiter))
                        doc_index = doc_line
                    self.signatures[doc_index] = self.hash_document(tokens)
                self.line_to_index[doc_line] = doc_line
                doc_line += 1
                if doc_line-doc_id_0 >= max_lines:
                    break

    def hash_document(self, document):
        """
        MinHash signature of a single document, does not add to dataset
        :param document: Set of tokens
        :return signature: numpy vector of MinHash signature
        """
        signature = np.empty(self._number_hash_functions, dtype=np.uint64)
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
        hv = int(sha1(token.encode('utf-8')).hexdigest(), 16) % (10 ** 12)
        # Do Carter and Wegman like hashing.
        values = np.bitwise_and((self._a * hv + self._b) % self._mersenne_prime, self._max_hash).astype(np.uint64)
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
    def __init__(self, number_hash_functions, threshold, number_threads=1):
        """
        :param number_hash_functions: Integer, number of hash functions
        :param threshold: Jaccard threshold in [0, 1]
        :param number_threads: For multiprocessing
        """
        self.pool = multiprocessing.Pool(number_threads)
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
        chunk_size = min(int(float(len(signatures))/self.pool._processes), 1000)
        function = partial(compute_bands, self._number_bands_per_doc)
        print 'Computing bands...'
        for doc_id, bands in self.pool.imap_unordered(function, signatures.iteritems(), chunk_size):
            if doc_id not in self.doc_to_bands:
                self.doc_to_bands[doc_id] = bands
            else:
                KeyError('Attempted to add same document multiple times')
            for band in bands:
                if band in self.band_to_docs:
                    self.band_to_docs[band].add(doc_id)
                else:
                    self.band_to_docs[band] = {doc_id}
            if doc_id % 1000 == 0:
                print 'Finished banding for doc ', str(doc_id)
        print 'Added ' + str(len(signatures)) + ' documents to the banding. Total of ' + str(self.number_bands) + ' bands with ' + str(self.number_docs_in_bands) + ' stored doc ids (including repeated elements in different bands.'

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


def compute_bands(number_bands_per_doc, docid_signature):
    """
    Compute bands of a signature
    :param signature: numpy vector of a single document's minhash signature
    :param number_bands_per_doc:
    :return bands: List of document bands
    """
    docid = docid_signature[0]
    signature = docid_signature[1]
    bands = set()
    for i, raw_band in enumerate(np.array_split(signature, number_bands_per_doc)):
        band = sha1("ab" + str(raw_band) + "ba" + str(i)).digest()
        bands.add(band)
    return docid, bands


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None  # if func_name is classmethod
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
