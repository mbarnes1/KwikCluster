from KwikCluster import consensus_clustering


def main():
    ## Synthetic Example
    clustering1 = frozenset([
        frozenset([1, 2, 3, 4]),
        frozenset([5, 6, 7])
    ])
    clustering2 = frozenset([
        frozenset([1, 2, 3]),
        frozenset([4, 5, 6, 7])
    ])
    clustering_combined = consensus_clustering([clustering1, clustering2])
    print 'Consensus clustering: ', clustering_combined


def links_to_labels(links):
    """
    :param links: Dict of [doc_id, set of connected doc ids]
    :return labels: Dict of [doc_id, label]
    """
    labels = dict()
    label = 1
    for (pivot, linkage) in links.iteritems():
        if pivot not in labels:
            labels[pivot] = label
            for id in linkage:
                labels[id] = label
            label += 1
    return labels


def links_to_sets(links):
    """
    :param links: Dict of [doc_id, set of connected doc ids]
    :return clusters: Frozen set of frozen sets, each frozen set contains doc ids in that cluster
    """
    removed = set()
    clusters = []
    for (pivot, linkage) in links.iteritems():
        if pivot not in removed:
            clusters.append(frozenset(linkage))
        removed.update(linkage)
        removed.add(pivot)
    clusters = frozenset(clusters)
    return clusters

if __name__ == '__main__':
    main()