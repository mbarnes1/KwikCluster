import cProfile
from MinHash import MinHash
from MinHash import Banding
from KwikCluster import kwik_cluster_minhash


def main():
    number_hash_functions = 200
    threshold = 0.9
    file_name = 'test/synthetic100000.txt'
    header_lines = 0
    number_processes = 4

    minhash = MinHash(number_hash_functions)
    minhash.hash_corpus(file_name, headers=header_lines, number_threads=number_processes, max_lines=5000)
    bands = Banding(number_hash_functions, threshold, number_threads=number_processes)
    bands.add_signatures(minhash.signatures)
    clusters = kwik_cluster_minhash(minhash, bands, threshold)
    print 'Finished with ', str(len(clusters)), ' clusters'


if __name__ == '__main__':
    cProfile.run('main()')