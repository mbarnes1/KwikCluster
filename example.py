import cProfile
from MinHash import MinHash
from MinHash import Banding
from KwikCluster import kwik_cluster


def main():
    number_hash_functions = 200
    threshold = 0.9
    file_name = 'test/synthetic.txt'
    header_lines = 0

    minhash = MinHash(number_hash_functions)
    minhash.hash_corpus(file_name, headers=header_lines)
    bands = Banding(number_hash_functions, threshold)
    bands.add_signatures(minhash.signatures)
    clusters = kwik_cluster(minhash, bands, threshold)
    print clusters


if __name__ == '__main__':
    cProfile.run('main()')