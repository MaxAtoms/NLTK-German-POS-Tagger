#!/bin/bash

pipenv run python ./german_tagger.py --data 1 --iter 1 --percentage 1
pipenv run python ./german_tagger.py --data 2 --iter 1 --percentage 80
pipenv run python ./german_tagger.py --data 3 --iter 1 --percentage 90

pipenv run python ./evaluator.py --data 1 --model 1
pipenv run python ./evaluator.py --data 2 --model 1
pipenv run python ./evaluator.py --data 3 --model 1

