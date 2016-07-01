import unittest
from CorrelationClustering.MinHash import MinHash, Banding
import numpy as np
from copy import deepcopy
import timeit
from draw_synthetic import draw_synthetic
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.number_hash_functions = 200
        self.threshold = 0.8
        self.minhash = MinHash(self.number_hash_functions)
        self.banding = Banding(self.number_hash_functions, self.threshold, number_threads=2)

    def test_hash_token(self):
        h1 = self.minhash._hash_token('hello')
        self.assertEqual(h1.shape, (self.number_hash_functions,))
        h2 = self.minhash._hash_token('world')
        self.assertEqual((h1 == h2).all(), False)
        h3 = self.minhash._hash_token('hello')
        np.testing.assert_array_equal(h1, h3)

    def test_hash_document(self):
        doc1 = frozenset('And under the boughs unbowed. All clothed in a snowy shroud. She had no heart so hardened. All under the boughs unbowed.'.split(' '))
        doc2 = frozenset('Each feather it fell from skin. Till threadbare while she grew thin. How were my eyes so blinded? Each feather it fell from skin.'.split(' '))
        doc3 = frozenset('And under the boughs unbowed. All clothed in a snowy shroud. She had no heart so hardened. All under the boughs unbowed.'.split(' '))
        self.assertEqual(self.minhash.hash_document(doc3).shape, (self.number_hash_functions,))
        self.assertEqual((self.minhash.hash_document(doc1) == self.minhash.hash_document(doc2)).all(), False)
        np.testing.assert_array_equal(self.minhash.hash_document(doc1), self.minhash.hash_document(doc3))

    def test_hash_corpus(self):
        self.minhash.hash_corpus('test/cranewife.txt', headers=1)
        doc0 = frozenset('And under the boughs unbowed. All clothed in a snowy shroud. She had no heart so hardened. All under the boughs unbowed.'.split(' '))
        doc1 = frozenset('Each feather it fell from skin. Till threadbare while she grew thin. How were my eyes so blinded? Each feather it fell from skin.'.split(' '))
        doc2 = doc0
        np.testing.assert_array_equal(self.minhash.signatures[0], self.minhash.hash_document(doc0))
        np.testing.assert_array_equal(self.minhash.signatures[1], self.minhash.hash_document(doc1))
        np.testing.assert_array_equal(self.minhash.signatures[2], self.minhash.hash_document(doc2))
        self.assertEqual(len(self.minhash.signatures), 3)

    def test_hash_corpus_list(self):
        number_threads = 4
        number_records = 100
        minhash1 = deepcopy(self.minhash)
        minhash2 = deepcopy(self.minhash)
        _ = draw_synthetic(number_records, 10)
        minhash1.hash_corpus('test/synthetic.txt', number_threads=1)
        with open('test/synthetic.txt') as ins:
            documents = [line for line in ins]
        minhash2.hash_corpus_list(documents, number_threads=number_threads)
        self.assertEqual(len(minhash1.signatures), len(minhash2.signatures))
        for key, value in minhash1.signatures.iteritems():
            print 'Testing doc ' + str(key)
            np.testing.assert_array_equal(value, minhash2.signatures[key])

    def test_max_lines(self):
        self.minhash.hash_corpus('test/synthetic.txt', headers=1, max_lines=12)
        self.assertEqual(len(self.minhash.signatures), 12)

    def test_corpus_multiprocessing(self):
        number_threads = 10
        number_records = 1000
        number_tests = 1
        minhash1 = deepcopy(self.minhash)
        minhash2 = deepcopy(self.minhash)
        _ = draw_synthetic(number_records, 50, output='test/synthetic.txt')
        t = timeit.Timer(lambda: minhash1.hash_corpus('test/synthetic.txt', number_threads=1))
        duration_single = t.timeit(number=number_tests)
        t = timeit.Timer(lambda: minhash2.hash_corpus('test/synthetic.txt', number_threads=number_threads))
        duration_multi = t.timeit(number=number_tests)
        self.assertEqual(len(minhash1.signatures), len(minhash2.signatures))
        for key, value in minhash1.signatures.iteritems():
            np.testing.assert_array_equal(value, minhash2.signatures[key])
        print 'Single process hashing time: ' + str(duration_single)
        print str(number_threads) + '-process hashing time: ' + str(duration_multi)

    def test_jaccard(self):
        doc1 = frozenset(['s'+str(i) for i in range(1, 1000)])
        doc2 = frozenset(['s'+str(i) for i in range(300, 1100)])
        sig1 = self.minhash.hash_document(doc1)
        sig2 = self.minhash.hash_document(doc2)
        self.minhash.signatures[0] = sig1
        self.minhash.signatures[1] = sig2
        j = self.minhash.jaccard(0, 1)
        self.assertAlmostEqual(j, 701.0/1100, delta=0.05)
        doc1 = frozenset(['38445', '90539', '23165', '99445', '26330', '93673', '98658', '47674', '22856', '19105', '6344', '7772', '47715', '22134', '76371', '27007', '25624', '30634', '16109', '97286'])
        doc2 = frozenset(['38446', '90539', '23166', '99445', '26331', '93673', '98659', '47674', '22857', '19105', '6345', '7772', '47716', '22134', '76372', '27007', '25626', '30634', '16110', '97286'])
        sig1 = self.minhash.hash_document(doc1)
        sig2 = self.minhash.hash_document(doc2)
        self.minhash.signatures[0] = sig1
        self.minhash.signatures[1] = sig2
        j = self.minhash.jaccard(0, 1)
        self.assertAlmostEqual(j, 10./30, delta=0.05)

    def test_add_signatures(self):
        number_tests = 1
        number_threads = 4
        number_records = 100
        _ = draw_synthetic(number_records, 20)
        self.minhash.hash_corpus('test/synthetic.txt', headers=1, number_threads=5)
        banding1 = Banding(self.number_hash_functions, self.threshold, number_threads=1)
        banding2 = Banding(self.number_hash_functions, self.threshold, number_threads=number_threads)
        t = timeit.Timer(lambda: banding1.add_signatures(self.minhash.signatures))
        duration_single = t.timeit(number=number_tests)
        t = timeit.Timer(lambda: banding2.add_signatures(self.minhash.signatures))
        duration_multi = t.timeit(number=number_tests)
        self.assertEqual(len(banding1.doc_to_bands), len(self.minhash.signatures))
        for key, value in banding1.band_to_docs.iteritems():
            self.assertSetEqual(value, banding2.band_to_docs[key])
        for key, value in banding1.doc_to_bands.iteritems():
            self.assertSetEqual(value, banding2.doc_to_bands[key])
        print 'Single process banding time: ' + str(duration_single)
        print str(number_threads) + '-process banding time: ' + str(duration_multi)

    #def test_calculate_bandwidth(self):