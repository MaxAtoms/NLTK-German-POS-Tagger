# NLTK-German-POS-Tagger

## Requirements

Make sure you have a working Python and [Pipenv](https://pipenv.pypa.io/en/latest/) installation.

## Setup

In the project directory, run `pipenv install`.
Obtain the TIGER Korpus:

```
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/download/tigercorpus-2.2.conll09.tar.gz -P corpora/
tar -xzvf corpora/tigercorpus-2.2.conll09.tar.gz -C corpora/
```

## Training

Run: `pipenv run python ./german_tagger.py`

## Evaluation

Run: `pipenv run python ./evaluator.py`

