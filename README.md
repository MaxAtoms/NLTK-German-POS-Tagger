# NLTK-German-POS-Tagger

Our final project report can be found [here](https://github.com/MaxAtoms/NLTK-German-POS-Tagger/blob/b9948d383f2089e7f55405b85d705f788e2dced5/report.pdf).
Results are automatically generated via a GitHub Action and published in [Releases](https://github.com/MaxAtoms/NLTK-German-POS-Tagger/releases/).

## Running directly on your machine

Make sure you have a working GNU Parallel, Python and [Pipenv](https://pipenv.pypa.io/en/latest/) installation. Execute `./entrypoint.sh`.

## Manual Training

More info: `pipenv run python ./german_tagger.py --help`

## Manual Evaluation

More info: `pipenv run python ./evaluator.py --help`

## Execution in Docker

- Build the image: `docker build -t pos-tagger:latest .`
- Run a container in this repository: `docker run -it -v $(pwd):/tmp pos-tagger:latest`
- Opening a shell for manual experiments: `docker run -it -v $(pwd):/tmp --entrypoint /bin/bash pos-tagger:latest`
