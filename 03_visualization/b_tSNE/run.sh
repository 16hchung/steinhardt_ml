#!/bin/bash

rm -f data/* # Clean output directories.

# Iterate over lattice structures.
for perplexity in 10 50 200 500 1000
do
  echo ${perplexity}
  python3 compute.py ${perplexity} &
done
wait
