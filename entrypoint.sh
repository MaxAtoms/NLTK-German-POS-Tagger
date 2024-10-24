#!/bin/bash

pipenv install
./get_corpora.sh

./train_eval.sh
