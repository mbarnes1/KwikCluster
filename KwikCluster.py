from copy import deepcopy


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


