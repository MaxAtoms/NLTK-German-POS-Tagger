#!/bin/bash

set -e

print_dashes() {
  printf -- '-%.s' {1..58}
  echo
}

train() {
  print_dashes
  pipenv run python ./german_tagger.py --data "tiger_train1_val99" --iter 1 --percentage 1
  print_dashes
  pipenv run python ./german_tagger.py --data "tiger_train80_val20" --iter 1 --percentage 80
  print_dashes
  pipenv run python ./german_tagger.py --data "tiger_train90_val10" --iter 1 --percentage 90
}

evaluate() {
  print_dashes
  echo "Result for a test split of 1%/99%"
  pipenv run python ./evaluator.py --data "tiger_train1_val99" --model 1 --description "TIGER Korpus: training on 1%, validation on 99%"
  print_dashes
  echo "Result for a test split of 80%/20%"
  pipenv run python ./evaluator.py --data "tiger_train80_val20" --model 1 --description "TIGER Korpus: training on 80%, validation on 20%"
  print_dashes
  echo "Result for a test split of 90%/10%"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --model 1 --description "TIGER Korpus: training on 90%, validation on 10%"

  print_dashes
  echo "Result for training on TIGER Korpus and validation on literary texts"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --testfile "corpora/novelette.conll" \
    --model 1 --description "Training: 90% of TIGER Korpus, Validation: Literary texts"
  print_dashes
  echo "Result for training on TIGER Korpus and validation on Open Subtitles"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --testfile "corpora/opensubtitles.conll" \
    --model 1 --description "Training: 90% of TIGER Korpus, Validation: Open Subtitles"
  print_dashes
  echo "Result for training on TIGER Korpus and validation on Christian sermons"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --testfile "corpora/sermononline.conll" \
    --model 1 --description "Training: 90% of TIGER Korpus, Validation: Christian Sermons"
  print_dashes
  echo "Result for training on TIGER Korpus and validation on TED Talks"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --testfile "corpora/ted.conll" \
    --model 1 --description "Training: 90% of TIGER Korpus, Validation: TED Talks"
  print_dashes
  echo "Result for training on TIGER Korpus and validation on Wikipedia"
  pipenv run python ./evaluator.py --data "tiger_train90_val10" --testfile "corpora/wikipedia.conll"\
    --model 1 --description "Training: 90% of TIGER Korpus, Validation: Wikipedia"
}

train
evaluate
