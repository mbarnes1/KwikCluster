import cProfile
from MinHash import MinHash, JaccardMatchFunction
from MinHash import Banding
from KwikCluster import kwik_cluster


def main():
    number_hash_functions = 200
    threshold = 0.5
    file_name = 'test/synthetic.txt'
    number_processes = 4

    minhash = MinHash(number_hash_functions)
    doc_ids = set()
    with open(file_name) as ins:
        for counter, line in enumerate(ins):
            doc_ids.add(counter)
            tokens = line.split(' ')
            minhash.add_document(counter, tokens)
    minhash.finish()
    bands = Banding(number_hash_functions, threshold, number_processes=number_processes)
    bands.add_signatures(minhash.signatures)
    match_function = JaccardMatchFunction(minhash, bands).match_function
    clusters = kwik_cluster(match_function, doc_ids)
    print ('Finished with ', str(len(clusters)), ' clusters')


if __name__ == '__main__':
    cProfile.run('main()')
