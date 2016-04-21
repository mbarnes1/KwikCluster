import unittest
from KwikCluster import kwik_cluster
from KwikCluster import clean
from KwikCluster import clusters_to_labels
import MinHash
from draw_synthetic import draw_synthetic
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        number_clusters = 2
        self.number_records = 100
        number_hash_functions = 200
        self.threshold = 0.05
        dataset = 'test/synthetic.txt'
        _, self.labels = draw_synthetic(self.number_records, number_clusters, output=dataset)
        self.minhash = MinHash.MinHash(number_hash_functions)
        self.minhash.hash_corpus(dataset)
        self.banding = MinHash.Banding(number_hash_functions, self.threshold)
        self.banding.add_signatures(self.minhash.signatures)

    def test_clusters_to_labels(self):
        clusters = [[1, 2, 3], [8, 9, 10], [5, 6]]
        labels = clusters_to_labels(clusters)
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[1], labels[3])
        self.assertEqual(labels[8], labels[9])
        self.assertEqual(labels[8], labels[10])
        self.assertEqual(labels[5], labels[6])
        self.assertNotEqual(labels[1], labels[8])
        self.assertNotEqual(labels[1], labels[5])

    def test_clean(self):
        self.assertEqual(len(self.banding.doc_to_bands), self.number_records)
        self.assertIn(3, self.banding.doc_to_bands)
        clean(self.banding, 3)
        self.assertEqual(len(self.banding.doc_to_bands), self.number_records-1)
        self.assertNotIn(3, self.banding.doc_to_bands)
        for band, doc_ids in self.banding.band_to_docs.iteritems():
            self.assertNotIn(3, doc_ids)

    def test_kwikcluster(self):
        print self.labels
        predicted_clusters = kwik_cluster(self.minhash, self.banding, self.threshold)
        true_clusters = dict()
        for doc_id, label in self.labels.iteritems():
            if label in true_clusters:
                true_clusters[label].add(doc_id)
            else:
                true_clusters[label] = {doc_id}
        true_clusters = frozenset([frozenset(docs) for _, docs in true_clusters.iteritems()])
        self.assertEqual(predicted_clusters, true_clusters)
