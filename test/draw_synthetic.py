import numpy as np


def draw_synthetic(number_records, number_clusters, output='synthetic.txt'):
    """
    Synthetic datset
    :param number_records:
    :param number_clusters:
    :return records: Dictionary of [doc id, text] where text is a string
    :return labels: Dictionary of [doc id, label] where label is an int
    """
    number_features = 20
    cluster_features = np.random.randint(0, 99999, size=(number_clusters, number_features))
    records = dict()
    labels = dict()
    with open(output, 'w') as ins:
        for record_id in range(0, number_records):
            cluster_id = np.random.randint(0, number_clusters)
            noise = np.random.randint(0, 2, size=(number_features,))
            record_features = cluster_features[cluster_id, :] + noise
            record_features = [str(f) for f in record_features]
            record_text = ' '.join(record_features)
            records[record_id] = record_text
            ins.write(record_text+'\n')
            labels[record_id] = cluster_id
        return records, labels
