#!/bin/bash
set -e

# Work-around for GitHub action issue :(
pip install --user pipenv

pipenv install
./get_corpora.sh

./train_eval.sh
