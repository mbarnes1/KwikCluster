import unittest
from MinHash import MinHash
import numpy as np
from MinHash import Banding
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.number_hash_functions = 200
        threshold = 0.8
        self.minhash = MinHash(self.number_hash_functions)
        self.banding = Banding(self.number_hash_functions, threshold)

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
        self.assertEqual(self.minhash._hash_document(doc3).shape, (self.number_hash_functions,))
        self.assertEqual((self.minhash._hash_document(doc1) == self.minhash._hash_document(doc2)).all(), False)
        np.testing.assert_array_equal(self.minhash._hash_document(doc1), self.minhash._hash_document(doc3))

    def test_hash_corpus(self):
        self.minhash.hash_corpus('cranewife.txt', headers=1)
        doc1 = frozenset('And under the boughs unbowed. All clothed in a snowy shroud. She had no heart so hardened. All under the boughs unbowed.'.split(' '))
        doc2 = frozenset('Each feather it fell from skin. Till threadbare while she grew thin. How were my eyes so blinded? Each feather it fell from skin.'.split(' '))
        doc3 = doc1
        np.testing.assert_array_equal(self.minhash.signatures[1], self.minhash._hash_document(doc1))
        np.testing.assert_array_equal(self.minhash.signatures[2], self.minhash._hash_document(doc2))
        np.testing.assert_array_equal(self.minhash.signatures[3], self.minhash._hash_document(doc3))
        self.assertEqual(len(self.minhash.signatures), 3)

    def test_jaccard(self):
        doc1 = frozenset(['s'+str(i) for i in range(1, 1000)])
        doc2 = frozenset(['s'+str(i) for i in range(300, 1100)])
        sig1 = self.minhash._hash_document(doc1)
        sig2 = self.minhash._hash_document(doc2)
        self.minhash.signatures[1] = sig1
        self.minhash.signatures[2] = sig2
        j = self.minhash.jaccard(1, 2)
        self.assertAlmostEqual(j, 701.0/1100, delta=0.05)

    def test_add_signature(self):
        self.minhash.hash_corpus('cranewife.txt', headers=1)
        self.banding._add_signature(7, self.minhash.signatures[1])
        self.banding._add_signature(11, self.minhash.signatures[3])
        self.assertEqual(len(self.banding._doc_to_bands), 2)
        self.assertEqual(7 in self.banding._band_to_docs[self.banding._doc_to_bands[7].pop()], True)

    def test_add_signatures(self):
        self.minhash.hash_corpus('cranewife.txt', headers=1)
        self.banding.add_signatures(self.minhash.signatures)
        self.assertEqual(len(self.banding._doc_to_bands), 3)
        self.assertEqual(1 in self.banding._band_to_docs[self.banding._doc_to_bands[1].pop()], True)
        self.assertEqual(2 in self.banding._band_to_docs[self.banding._doc_to_bands[2].pop()], True)

    #def test_calculate_bandwidth(self):