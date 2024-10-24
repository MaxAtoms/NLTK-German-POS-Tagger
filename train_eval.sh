#!/bin/bash

set -e

print_dashes() {
  printf -- '-%.s' {1..58}
  echo
}

print_dashes
pipenv run python ./german_tagger.py --data 1 --iter 1 --percentage 1
print_dashes
pipenv run python ./german_tagger.py --data 2 --iter 1 --percentage 80
print_dashes
pipenv run python ./german_tagger.py --data 3 --iter 1 --percentage 90

print_dashes
echo "Result for a test split of 1%/99%"
pipenv run python ./evaluator.py --data 1 --model 1
print_dashes
echo "Result for a test split of 80%/20%"
pipenv run python ./evaluator.py --data 2 --model 1
print_dashes
echo "Result for a test split of 90%/10%"
pipenv run python ./evaluator.py --data 3 --model 1

