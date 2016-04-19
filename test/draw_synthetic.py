import numpy as np


def draw_synthetic(number_records, number_clusters):
    """
    Synthetic datset
    :param number_records:
    :param number_clusters:
    :return records: Dictionary of [doc id, text] where text is a string
    :return labels: Dictionary of [doc id, label] where label is an int
    """
    number_features = 10
    cluster_features = np.random.randint(0, 99, size=(number_clusters, number_features))
    records = dict()
    labels = dict()
    for record_id in range(1, number_records):
        cluster_id = np.random.randint(1, number_clusters)
        noise = np.random.randint(-5, 5, size=(1, number_features))
        record_features = cluster_features[cluster_id, :] + noise
        record_features = [str('s' + f) for f in record_features]
        record_text = ' '.join(record_features)
        records[record_id] = record_text
        labels[record_id] = cluster_id
    return records, labels
