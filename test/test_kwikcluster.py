import unittest
from CorrelationClustering.KwikCluster import kwik_cluster, clusters_to_labels, consensus_clustering, JaccardMatchFunction
from CorrelationClustering import MinHash
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
        with open(dataset, 'rb') as ins:
            for line_number, line in enumerate(ins):
                tokens = line.split(' ')
                self.minhash.add_document(line_number, tokens)
        self.minhash.finish()
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

    def test_kwikcluster_minhash(self):
        match_function = JaccardMatchFunction(self.minhash, self.banding).match_function
        doc_ids = set(self.minhash.signatures.keys())
        predicted_clusters = kwik_cluster(match_function, doc_ids)
        true_clusters = dict()
        for doc_id, label in self.labels.iteritems():
            if label in true_clusters:
                true_clusters[label].add(doc_id)
            else:
                true_clusters[label] = {doc_id}
        true_clusters = frozenset([frozenset(docs) for _, docs in true_clusters.iteritems()])
        self.assertEqual(predicted_clusters, true_clusters)

    def test_consensus_clustering(self):
        clustering1 = frozenset([frozenset([1, 2, 3, 4]), frozenset([5, 6, 7])])
        clustering2 = frozenset([frozenset([1, 2, 3]), frozenset([4, 5, 6, 7])])
        clustering3 = frozenset([frozenset([1, 2]), frozenset([3, 4, 5, 6, 7])])
        clusterings = [clustering1, clustering2, clustering3]
        clustering = consensus_clustering(clusterings)
        n_matches = 0
        for cluster in clustering:
            if (1 in cluster and 2 in cluster) or (5 in cluster and 6 in cluster and 7 in cluster):
                n_matches += 1
        self.assertEqual(n_matches, 2)

