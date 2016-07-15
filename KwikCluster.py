import cProfile
from MinHash import MinHash, Banding, JaccardMatchFunction
import sys
import getopt
from numpy import Inf
from numpy import random
from itertools import izip
__author__ = 'Matt Barnes'


def main(argv):
    """
    Cluster a plain text file, one document per line with unigram tokens.
    :param argv: See below
    :return:
    """
    number_hash_functions = 200
    threshold = 0.9
    number_processes = 1
    max_lines = Inf
    helpline = 'KwikCluster.py -i <inputfile> -o <outputfile> -t <threshold> -f <numberhashfunctions> -c <numberprocesses> -m <maxlines>'
    try:
        opts, args = getopt.getopt(argv, "h:i:o:t:f:p:m:", ["ifile=", "ofile=", "threshold=", "hashfunctions=", "processes=", "maxlines="])
    except getopt.GetoptError:
        print helpline
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpline
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        elif opt in ("-f", "--hashfunctions"):
            number_hash_functions = int(arg)
        elif opt in ("-p", "--processes"):
            number_processes = int(arg)
        elif opt in ("-m", "--maxlines"):
            max_lines = int(arg)
    minhash = MinHash(number_hash_functions)
    bands = Banding(number_hash_functions, threshold, number_processes=number_processes)
    doc_ids_to_cluster = set()
    with open(input_file, 'rb') as ins:
        for line_number, line in enumerate(ins):
            if line_number % 1000 == 0:
                print 'Reading in document ' + str(line_number)
            if line_number > max_lines:
                break
            doc_ids_to_cluster.add(line_number)
            tokens = line.split(' ')
            minhash.add_document(line_number, tokens)
    minhash.finish()
    bands.add_signatures(minhash.signatures)
    match_function = JaccardMatchFunction(minhash, bands).match_function
    clusters = kwik_cluster(match_function, doc_ids_to_cluster)
    print 'Finished clustering. Found ', str(len(clusters)), ' clusters'
    with open(output_file, 'w') as ins:
        for cluster in clusters:
            line = ' '.join([str(doc_index) for doc_index in cluster])
            ins.write(line + '\n')


def kwik_cluster(match_function, doc_indices):
    """
    KwikCluster (Ailon et al. 2008), with edges between any docs with at least one "feature"
    :param match_function: Function handle. match_function(pivot_doc_index) returns set of all doc_indices with edge to pivot_doc_index
    :param doc_indices: Set of doc indices to cluster
    :return clusters: Frozen set of frozen sets, each subset contains doc ids in that cluster
    """
    print 'Running KwikCluster on documents...'
    clusters = set()
    while doc_indices:
        if len(clusters) % 100 == 0:
            print '    KwikCluster on remaining ' + str(len(doc_indices)) + ' documents'
        pivot_index = doc_indices.pop()
        cluster = match_function(pivot_index)
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
        probs = float(len(self.list_id_to_matches))*random.uniform(size=len(potential_matches))
        matches = set([doc_id])
        for prob, (potential_match, count) in izip(probs, potential_matches.iteritems()):
            if float(count)/float(len(self.list_id_to_matches)) > prob:
                matches.add(potential_match)
        matches.add(doc_id)
        return matches


def consensus_clustering(clusterings):
    """
    Consensus Clustering with KwikCluster (Ailon et al. 2008). An 11/7 approximation algorithm, in linear time
    :param clusterings: List of clusterings, where each clustering is a frozen set of clusters (each cluster is a frozen set of doc ids)
    :return clusters: Frozen set of frozen sets, each subset contains doc ids in that cluster
    """
    doc_ids = set()
    for clustering in clusterings:
        for cluster in clustering:
            doc_ids.update(set(cluster))
    match_function = ConsensusClusteringMatchFunction(clusterings).match_function
    clusters = kwik_cluster(match_function, doc_ids)
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
    cProfile.run('main(sys.argv[1:])')


