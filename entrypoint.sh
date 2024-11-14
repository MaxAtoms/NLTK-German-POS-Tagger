#!/bin/bash
set -e

# Work-around for GitHub Action problem :(
if [ "$GITHUB_ACTIONS" == "true" ]; then
  pip install --user pipenv
fi

pipenv install
./get_corpora.sh
./training.sh
./evaluation.sh
