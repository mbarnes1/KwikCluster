from draw_synthetic import draw_synthetic
from KwikCluster import kwik_cluster, clusters_to_labels, consensus_clustering, JaccardMatchFunction, ConsensusClusteringMatchFunction
from MinHash import MinHash, Banding
import Queue
import unittest
__author__ = 'mbarnes1'


class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

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
        number_clusters = 2
        number_records = 100
        number_hash_functions = 200
        threshold = 0.05
        dataset = 'synthetic.txt'
        _, labels = draw_synthetic(number_records, number_clusters, output=dataset)
        minhash = MinHash(number_hash_functions)
        with open(dataset, 'rb') as ins:
            for line_number, line in enumerate(ins):
                tokens = line.split(' ')
                minhash.add_document(line_number, tokens)
        minhash.finish()
        banding = Banding(number_hash_functions, threshold)
        banding.add_signatures(minhash.signatures)

        match_function = JaccardMatchFunction(minhash, banding).match_function
        doc_ids = set(minhash.signatures.keys())
        predicted_clusters = kwik_cluster(match_function, doc_ids)
        true_clusters = dict()
        for doc_id, label in labels.iteritems():
            if label in true_clusters:
                true_clusters[label].add(doc_id)
            else:
                true_clusters[label] = {doc_id}
        true_clusters = frozenset([frozenset(docs) for _, docs in true_clusters.iteritems()])
        self.assertEqual(predicted_clusters, true_clusters)

    def test_consensus_match_function(self):
        clustering1 = frozenset([frozenset([1, 2, 3]), frozenset([4, 5])])
        clustering2 = frozenset([frozenset([1, 2]), frozenset([3, 4, 5])])
        clusterings = [clustering1, clustering2]
        match_function = ConsensusClusteringMatchFunction(clusterings).match_function
        self.assertTrue(any([match_function(1) == frozenset([1, 2]), match_function(1) == frozenset([1, 2, 3])]))
        self.assertTrue(any([match_function(5) == frozenset([4, 5]), match_function(5) == frozenset([3, 4, 5])]))

    def test_consensus_clustering(self):
        clustering1 = frozenset([frozenset([1, 2, 3]), frozenset([4, 5])])
        clustering2 = frozenset([frozenset([1, 2]), frozenset([3, 4, 5])])
        clusterings = [clustering1, clustering2]
        queue = Queue.Queue()
        queue.put(1)
        queue.put(4)
        clustering = consensus_clustering(clusterings, seed_queue=queue)
        self.assertTrue(any([
            frozenset([1, 2]) in clustering,
            frozenset([1, 2, 3]) in clustering
        ]))
        self.assertTrue(any([
            frozenset([4, 5]) in clustering,
            frozenset([3, 4, 5]) in clustering
        ]))
        all_ids = set()
        for cluster in clustering:
            all_ids.update(cluster)
        self.assertEqual(all_ids, {1, 2, 3, 4, 5})

