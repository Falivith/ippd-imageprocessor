#!/bin/bash

make clean
make compile
for num in {1..6}; do
    for ((i=1; i<=30; i++))
    do
        mpirun -np $num ./main
    done
done
make clean