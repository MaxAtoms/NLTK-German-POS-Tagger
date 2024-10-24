#!/bin/bash
set -e

# Work-around for GitHub action issue :(
pip install --user pipenv

echo $(who)
echo $(id)
echo $(pip show pipenv)

pipenv install
./get_corpora.sh

./train_eval.sh
