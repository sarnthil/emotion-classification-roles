#!/bin/bash

# Activate the virtualenv
source transformers/bin/activate

for mask in only without inband
do
    for dataset in eca es
    do
        # just cause
        echo python roberta_classification_simple.py -d$dataset -m$mask -rcause
        python roberta_classification_simple.py -d$dataset -m$mask -rcause
        rm -rf outputs
    done
    for dataset in et gne reman
    do
        for role in cause cue target experiencer
        do
            echo python roberta_classification_simple.py -d$dataset -m$mask -r$role
            python roberta_classification_simple.py -d$dataset -m$mask -r$role
            rm -rf outputs
        done
    done
done

for dataset in eca es et gne reman  # all
do
    echo python roberta_classification_simple.py -d$dataset -mall
    python roberta_classification_simple.py -d$dataset -mall
    rm -rf outputs
done
