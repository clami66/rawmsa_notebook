#!/bin/bash

# only for reproducibility and to remember what we run to cluster. Not a script you should run
mmseqs easy-cluster disprot.fasta clustering/disprot /tmp/ --min-seq-id 0.3 -c 0.8 --cov-mode 1
