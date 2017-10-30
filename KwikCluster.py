import argparse
from itertools import izip
from MinHash import MinHash, Banding, JaccardMatchFunction
from numpy import Inf, random
import sys


__author__ = 'Matt Barnes'


def main(argv):
    """
    Cluster a plain text file, one document per line with unigram tokens.
    :param argv: See below
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path",
                        type=str,
                        help="Path to text file to cluster. One document per line.")

    parser.add_argument("output_file_path",
                        type=str,
                        help="Path to output cluster results. One cluster per line, with space-delimited cluster "
                             "members referenced according to zero-indexed line number in input-file-path.")

    parser.add_argument("--threshold",
                        type=float,
                        default=0.9,
                        help="Jaccard score cutoff threshold for a match between two documents.")

    parser.add_argument("--number-hash-functions",
                        type=int,
                        default=200,
                        help="Jaccard score cutoff threshold for a match between two documents.")

    parser.add_argument("--number-processes",
                        type=int,
                        default=1,
                        help="Number of parallel processes for hashing documents.")

    parser.add_argument("--max-lines",
                        type=int,
                        default=Inf,
                        help="Maximum number of lines to read from input-file-path.")

    args = parser.parse_args(argv)

    kwik_cluster_text_file(args)


def kwik_cluster_text_file(args):
    minhash = MinHash(args.number_hash_functions)
    bands = Banding(args.number_hash_functions, args.threshold, number_processes=args.number_processes)
    doc_ids_to_cluster = set()
    with open(args.input_file_path, 'rb') as ins:
        for line_number, line in enumerate(ins):
            if line_number % 1000 == 0:
                print 'Reading in document ' + str(line_number)
            if line_number > args.max_lines:
                break
            doc_ids_to_cluster.add(line_number)
            tokens = line.split(' ')
            minhash.add_document(line_number, tokens)
    minhash.finish()
    bands.add_signatures(minhash.signatures)
    match_function = JaccardMatchFunction(minhash, bands).match_function
    clusters = kwik_cluster(match_function, doc_ids_to_cluster)
    print 'Finished clustering. Found ', str(len(clusters)), ' clusters'
    with open(args.output_file_path, 'w') as ins:
        for cluster in clusters:
            line = ' '.join([str(doc_index) for doc_index in cluster])
            ins.write(line + '\n')


def kwik_cluster(match_function, doc_indices, seed_queue=None):
    """
    KwikCluster (Ailon et al. 2008), with edges between any docs with at least one "feature"
    :param match_function: Function handle. match_function(pivot_doc_index) returns set of all doc_indices with edge to pivot_doc_index
    :param doc_indices: Set of doc indices to cluster
    :param seed_queue: [Queue] Pop indices in this order (if possible). If none, pop randomly
    :return clusters: Frozen set of frozen sets, each subset contains doc ids in that cluster
    """
    print 'Running KwikCluster on documents...'
    clusters = set()
    while doc_indices:
        if len(clusters) % 100 == 0:
            print '    KwikCluster on remaining ' + str(len(doc_indices)) + ' documents'
        while seed_queue and not seed_queue.empty():
            pivot_index = seed_queue.get()
            if pivot_index in doc_indices:
                break
        else:  # seed is empty
            pivot_index = doc_indices.pop()
        cluster = match_function(pivot_index).intersection(doc_indices)
        cluster.add(pivot_index)
        doc_indices.difference_update(cluster)
        clusters.add(frozenset(cluster))
    clusters = frozenset(clusters)
    print 'Clustered into ' + str(len(clusters)) + ' clusters'
    return clusters


class ConsensusClusteringMatchFunction(object):
    """
    Probabilistic match function, based on fraction of links across all clusterings
    """
    def __init__(self, clusterings):
        self.list_id_to_matches = []
        for clustering in clusterings:
            id_to_matches = dict()
            for cluster in clustering:
                for doc_id in cluster:
                    id_to_matches[doc_id] = cluster
            self.list_id_to_matches.append(id_to_matches)

    def match_function(self, doc_id):
        """
        Returns all matching doc ids, probabilistically
        :param doc_id:
        :return matches: Set of matching doc ids (including doc_id
        """
        potential_matches = dict()
        for id_to_matches in self.list_id_to_matches:
            for potential_match in id_to_matches[doc_id]:
                if potential_match in potential_matches:
                    potential_matches[potential_match] += 1
                else:
                    potential_matches[potential_match] = 1
        probs = random.uniform(size=len(potential_matches))
        matches = set([doc_id])
        for prob, (potential_match, count) in izip(probs, potential_matches.iteritems()):
            if float(count)/float(len(self.list_id_to_matches)) > prob:
                matches.add(potential_match)
        matches.add(doc_id)
        return matches


def consensus_clustering(clusterings, seed_queue=None):
    """
    Consensus Clustering with KwikCluster (Ailon et al. 2008). An 11/7 approximation algorithm, in linear time
    :param clusterings: List of clusterings, where each clustering is a frozen set of clusters (each cluster is a frozen set of doc ids)
    :param seed_queue: [Queue] Pop indices in this order (if possible). If none, pop randomly
    :return clusters: Frozen set of frozen sets, each subset contains doc ids in that cluster
    """
    doc_ids = set()
    for clustering in clusterings:
        for cluster in clustering:
            doc_ids.update(set(cluster))
    match_function = ConsensusClusteringMatchFunction(clusterings)
    clusters = kwik_cluster(match_function.match_function, doc_ids, seed_queue=seed_queue)
    return clusters


def clusters_to_labels(clusters):
    """
    :param clusters: List of lists, each sublist contains doc ids in that cluster
    :return labels: Dict of [doc_id, cluster_label] where cluster_label are assigned from positive ints starting at 1
    """
    labels = dict()
    for label, cluster in enumerate(clusters):
        for doc_id in cluster:
            labels[doc_id] = label
    return labels


def clean(doc_to_features, feature_to_docs, doc_id):
    """
    Removes ID from all traces of bands
    :param doc_to_features: Dict mapping [doc, set of features]
    :param feature_to_docs: Dict mapping [feature, set of doc_ids]
    :param doc_id: Doc ID
    """
    features = doc_to_features.pop(doc_id)
    for feature in features:
        feature_to_docs[feature].remove(doc_id)


if __name__ == '__main__':
    main(sys.argv[1:])


