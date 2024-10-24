#!/bin/bash

set -e

print_dashes() {
  printf -- '-%.s' {1..58}
  echo
}

print_dashes
pipenv run python ./german_tagger.py --data "tiger_train1_val99" --iter 1 --percentage 1
print_dashes
pipenv run python ./german_tagger.py --data "tiger_train80_val20" --iter 1 --percentage 80
print_dashes
pipenv run python ./german_tagger.py --data "tiger_train90_val10" --iter 1 --percentage 90

print_dashes
echo "Result for a test split of 1%/99%"
pipenv run python ./evaluator.py --data "tiger_train1_val99" --model 1 --description "TIGER Korpus: training on 1%, validation on 99%"
print_dashes
echo "Result for a test split of 80%/20%"
pipenv run python ./evaluator.py --data "tiger_train80_val20" --model 1 --description "TIGER Korpus: training on 80%, validation on 20%"
print_dashes
echo "Result for a test split of 90%/10%"
pipenv run python ./evaluator.py --data "tiger_train90_val10" --model 1 --description "TIGER Korpus: training on 90%, validation on 10%"

