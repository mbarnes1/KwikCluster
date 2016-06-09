from KwikCluster import consensus_clustering
from KwikCluster import clusters_to_labels
import cPickle as pickle
from sklearn.metrics import mutual_info_score


def main():
    ## Synthetic Example
    clustering1 = {
        '1': {'1', '2', '3', '4'},
        '2': {'1', '2', '3', '4'},
        '3': {'1', '2', '3', '4'},
        '4': {'1', '2', '3', '4'},
        '5': {'5', '6', '7'},
        '6': {'5', '6', '7'},
        '7': {'5', '6', '7'},
    }
    clustering2 = {
        '1': {'1', '2', '3'},
        '2': {'1', '2', '3'},
        '3': {'1', '2', '3'},
        '4': {'4', '5', '6', '7'},
        '5': {'4', '5', '6', '7'},
        '6': {'4', '5', '6', '7'},
        '7': {'4', '5', '6', '7'},
    }
    labels1_dict = clusters_to_labels(links_to_sets(clustering1))
    labels2_dict = clusters_to_labels(links_to_sets(clustering2))
    clustering_combined = consensus_clustering([clustering1, clustering2])
    labels_combined_dict = clusters_to_labels(clustering_combined)
    labels1 = []
    labels2 = []
    labelsCombined = []
    for doc_id in range(1, len(labels1_dict) + 1):
        labels1.append(labels1_dict[str(doc_id)])
        labels2.append(labels2_dict[str(doc_id)])
        labelsCombined.append(labels_combined_dict[str(doc_id)])
    print('Mutual information between labels1 and labels2 is ' + str(mutual_info_score(labels1, labels2)))


    ## Real data
    phone_clusters = pickle.load(open('Consensus/WadePresentation/phone_cluster.pkl', 'rb'))
    text_clusters = pickle.load(open('Consensus/WadePresentation/text_cluster.pkl', 'rb'))
    phone_labels_dict = clusters_to_labels(links_to_sets(phone_clusters))
    text_labels_dict = clusters_to_labels(links_to_sets(text_clusters))
    phone_labels = []
    text_labels = []
    for doc_id in range(0, len(phone_labels_dict)):
        phone_labels.append(phone_labels_dict[str(doc_id)])
        text_labels.append(text_labels_dict[str(doc_id)])
    print('Mutual information between phone and text labels is ' + str(mutual_info_score(phone_labels, text_labels)))
    print('Mutual information between text and phone labels is ' + str(mutual_info_score(text_labels, phone_labels)))

    clustering_combined = consensus_clustering([phone_clusters, text_clusters])
    labels_combined_dict = clusters_to_labels(clustering_combined)
    labelsCombined = []
    for doc_id in range(1, len(labels1_dict) + 1):
        labelsCombined.append(labels_combined_dict[str(doc_id)])
    print('Mutual information between phone and consensus labels is ' + str(mutual_info_score(phone_labels, labelsCombined)))
    print('Mutual information between consensus and phone labels is ' + str(mutual_info_score(labelsCombined, phone_labels)))
    print('Mutual information between text and consensus labels is ' + str(mutual_info_score(text_labels, labelsCombined)))
    print('Mutual information between consensus and text labels is ' + str(mutual_info_score(labelsCombined, text_labels)))


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