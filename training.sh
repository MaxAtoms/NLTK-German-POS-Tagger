#!/bin/bash

set -e

dataset=(
  "--data=tiger_tagger --percentage=90"
  "--data=universal_tagger --percentage=100"
)

iterations=(
  "--iter=5 --model=2"
  "--iter=10 --model=3"
  "--iter=15 --model=4"
)

echo "Starting training..."

# Create the dataset and train a model for one iteration
parallel --joblog training-log1 eval pipenv run python german_tagger.py --train=1 --create=1 --model=1 --iter=1 {} ">" /dev/null ::: "${dataset[@]}"

# Train models with more iterations
parallel --joblog training-log2 eval pipenv run python german_tagger.py --train=1 --create=0 {} {} ">" /dev/null ::: "${dataset[@]}" ::: "${iterations[@]}"

