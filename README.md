# KwikCluster
KwikCluster [[1]](#ailon) using MinHash [[2]](#broder) as a match function.

## Basic usage with flat text files:
```
usage: KwikCluster.py [-h] [--threshold THRESHOLD]
                      [--number-hash-functions NUMBER_HASH_FUNCTIONS]
                      [--number-processes NUMBER_PROCESSES]
                      [--max-lines MAX_LINES]
                      input_file_path output_file_path

positional arguments:
  input_file_path       Path to text file to cluster. One document per line.
  output_file_path      Path to output cluster results. One cluster per line,
                        with space-delimited cluster members referenced
                        according to zero-indexed line number in input-file-
                        path.

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Jaccard score cutoff threshold for a match between two
                        documents. (default: 0.9)
  --number-hash-functions NUMBER_HASH_FUNCTIONS
                        Jaccard score cutoff threshold for a match between two
                        documents. (default: 200)
  --number-processes NUMBER_PROCESSES
                        Number of parallel processes for hashing documents.
                        (default: 1)
  --max-lines MAX_LINES
                        Maximum number of lines to read from input-file-path.
                        (default: inf)
```

## More than basic usage
For custom document feeding and match functions, see `example.py`.

## Consensus clustering
This package also implements *consensus clustering*, which combines multiple clusterings into a single clustering according to the objective in [[1]](#ailon). For an example usage, see `example_consensus.py`.

### References:
1. <a name="ailon"></a>Ailon, N., Charikar, M., & Newman, A. (2008). Aggregating inconsistent information. Journal of the ACM, 55(5),1–27. http://doi.org/10.1145/1411509.1411513
2. <a name="broder"></a>Broder, A. Z. (1997). On the resemblance and containment of documents. Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No.97TB100171), 1–9. http://doi.org/10.1109/SEQUEN.1997.666900
