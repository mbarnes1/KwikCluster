from draw_synthetic import draw_synthetic
from MinHash import MinHash, Banding
import numpy as np
import timeit
import unittest


__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.number_hash_functions = 200
        self.threshold = 0.8
        self.minhash = MinHash(self.number_hash_functions)
        self.banding = Banding(self.number_hash_functions, self.threshold, number_processes=2)

    def tearDown(self):
        self.minhash.finish()
        self.banding.close()

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
        with open('synthetic.txt', 'rb') as ins:
            for line_number, line in enumerate(ins):
                tokens = line.split(' ')
                self.minhash.add_document(line_number, tokens)
        banding1 = Banding(self.number_hash_functions, self.threshold, number_processes=1)
        banding2 = Banding(self.number_hash_functions, self.threshold, number_processes=number_threads)
        t = timeit.Timer(lambda: banding1.add_signatures(self.minhash.signatures))
        duration_single = t.timeit(number=number_tests)
        t = timeit.Timer(lambda: banding2.add_signatures(self.minhash.signatures))
        duration_multi = t.timeit(number=number_tests)
        banding1.close()
        banding2.close()
        self.assertEqual(len(banding1.doc_to_bands), len(self.minhash.signatures))
        for key, value in banding1.band_to_docs.iteritems():
            self.assertSetEqual(value, banding2.band_to_docs[key])
        for key, value in banding1.doc_to_bands.iteritems():
            self.assertSetEqual(value, banding2.doc_to_bands[key])
        print 'Single process banding time: ' + str(duration_single)
        print str(number_threads) + '-process banding time: ' + str(duration_multi)


    #def test_calculate_bandwidth(self):