# KwikCluster

python2.7 KwikCluster.py -i <inputfile> -o <outputfile> -d <numberheaderlines> -t <threshold> -f <numberhashfunctions> -c <numberthreads> -m <maxlines>

Inputs:
-i <inputfile> - Path to raw flat text file, with one sample per line
-d <numberheaderlines> - Number of header lines in the input file to skip. Default = 0
-m <maxlines> - Max number of lines to use from <inputfile>
-t <threshold> - Jaccard similarity cut-off threshold. Default = 0.9.
-f <numberhashfunctions> - Number of hash functions to use in MinHash approximation to the Jaccard coefficient. Default = 200.
-c <numberthreads> - Number of processes to run in parallel. Default = 1.

Outputs:
-o <outputfile> - Name of output file to create. Each line corresponds to a cluster, as a list of zero-indexed document line numbers in <inputfile>
