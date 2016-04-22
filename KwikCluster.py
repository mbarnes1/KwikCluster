# -*- coding: utf-8 -*-
from copy import deepcopy
import cProfile
from MinHash import MinHash
from MinHash import Banding
import sys
import getopt
from numpy import Inf


def main(argv):
    number_hash_functions = 200
    threshold = 0.9
    header_lines = 0
    number_threads = 1
    max_lines = Inf
    helpline = 'test.py -i <inputfile> -o <outputfile> -d <numberheaderlines> -t <threshold> -f <numberhashfunctions> -c <numberthreads> -m <maxlines>'
    try:
        opts, args = getopt.getopt(argv, "h:i:o:d:t:f:c:m:", ["ifile=", "ofile=", "headerlines=", "threshold=", "hashfunctions=", "threads=", "maxlines="])
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
        elif opt in ("-d", "--headerlines"):
            header_lines = int(arg)
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        elif opt in ("-f", "--hashfunctions"):
            number_hash_functions = int(arg)
        elif opt in ("-c", "--threads"):
            number_threads = int(arg)
        elif opt in ("-m", "--maxlines"):
            max_lines = int(arg)
    minhash = MinHash(number_hash_functions)
    minhash.hash_corpus(input_file, headers=header_lines, number_threads=number_threads, max_lines=max_lines)
    bands = Banding(number_hash_functions, threshold)
    bands.add_signatures(minhash.signatures, number_threads=number_threads)
    clusters = kwik_cluster(minhash, bands, threshold)
    with open(output_file, 'w') as ins:
        for cluster in clusters:
            line = ' '.join([str(doc_id) for doc_id in cluster])
            ins.write(line + '\n')


def kwik_cluster(minhash, bands_original, threshold, destructive=True):
    """
    KwikCluster (Ailon et al. 2008)
    :param minhash: MinHash object
    :param bands_original: Banding object
    :param threshold: Threshold to cluster at, >= bands.threshold
    :param destructive: Whether to destructively operate on bands (faster)
    :return clusters: Frozen set of frozen sets, each subset contains doc ids in that cluster
    """
    if threshold < bands_original.get_threshold():
        raise ValueError('Clustering threshold must be greater than or equal to threshold band threshold to find all matches with high probability')
    if destructive:
        bands = bands_original
    else:
        bands = deepcopy(bands_original)
    clusters = set()
    while bands.doc_to_bands:
        print 'KwikCluster on remaining ' + str(len(bands.doc_to_bands)) + ' documents'
        (pivot_id, pivot_bands) = bands.doc_to_bands.popitem()
        bands.doc_to_bands[pivot_id] = pivot_bands
        pivot_bands = deepcopy(pivot_bands)
        clean(bands, pivot_id)
        cluster = {pivot_id}
        for band in pivot_bands:
            for doc_id in deepcopy(bands.band_to_docs[band]):
                J = minhash.jaccard(pivot_id, doc_id)
                if J >= threshold:
                    cluster.add(doc_id)
                    clean(bands, doc_id)
        clusters.add(frozenset(cluster))
    clusters = frozenset(clusters)
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


def clean(bands, doc_id):
    """
    Removes ID from all traces of bands
    :param bands: Banding object
    :param doc_id: Doc ID
    """
    id_bands = bands.doc_to_bands.pop(doc_id)
    for band in id_bands:
        bands.band_to_docs[band].remove(doc_id)


if __name__ == '__main__':
    cProfile.run('main(sys.argv[1:])')


