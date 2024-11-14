#!/bin/bash

set -e

datasets=(
  "--testfile=corpora/novelette.conll"
  "--testfile=corpora/ted.conll"
  "--testfile=corpora/sermononline.conll"
  "--testfile=corpora/wikipedia.conll"
  "--testfile=corpora/opensubtitles.conll"
  )

models=(
  "--model=1"
  "--model=2"
  "--model=3"
  "--model=4"
  )

tagdict=(
  "--tagdict=0"
  "--tagdict=1"
  )

echo "Starting evaluation..."

# Evaluate 90/10 training-test-split on TIGER Korpus
parallel --joblog evaluation-log1 eval pipenv run python evaluator.py --data=tiger_tagger {} {} ">" /dev/null ::: "${models[@]}" ::: "${tagdict[@]}"

# Evaluate the fully trained model on different corpora
parallel --joblog evaluation-log2 eval pipenv run python evaluator.py --data=universal_tagger --percentage=100 {} {} ">" /dev/null ::: "${datasets[@]}" ::: "${models[@]}"
