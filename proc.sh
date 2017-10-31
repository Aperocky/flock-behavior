#!/bin/bash
for num in $(seq 25 25 100)
do
    echo "Getting "$num" Robot's chart and movie"
    python3 flock.py $num $1$num
done
