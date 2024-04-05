#!/bin/bash
set -eox pipefail

# accepts a list of cluster centers and a  .tsv cluster file as input, splits in 10 folds
repseqs=$1
clusters=$2
counter=0

# cleanup
rm -f $(dirname $repseqs)/fold_*

while read id; do
    echo $id
    fold=$(($counter % 10))

    # get all sequences in the cluster where seq "$id" is representative
    # using grep was giving unexpected results
    # members=$(grep "${id} " $clusters | awk '{ print $2 }')

    members=$(awk -v id=$id '$1 == id { print $2 }' $clusters)

    echo "$members" >> $(dirname $repseqs)/fold_${fold}

    counter=$((counter+1))
done<$repseqs
