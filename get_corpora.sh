#!/bin/bash

# TIGER Korpus
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/download/tigercorpus-2.2.conll09.tar.gz --no-verbose -P corpora/
tar -xzvf corpora/tigercorpus-2.2.conll09.tar.gz -C corpora/

# Gold Standard from "Evaluating Off-the-Shelf NLP Tools for German" (Ortmann et al., 2019)
wget https://raw.githubusercontent.com/rubcompling/konvens2019/refs/heads/master/data/gold/balanced/annotations/novelette.conll --no-verbose -P corpora/
wget https://raw.githubusercontent.com/rubcompling/konvens2019/refs/heads/master/data/gold/balanced/annotations/opensubtitles.conll --no-verbose -P corpora/
wget https://raw.githubusercontent.com/rubcompling/konvens2019/refs/heads/master/data/gold/balanced/annotations/sermononline.conll --no-verbose -P corpora/
wget https://raw.githubusercontent.com/rubcompling/konvens2019/refs/heads/master/data/gold/balanced/annotations/ted.conll --no-verbose -P corpora/
wget https://raw.githubusercontent.com/rubcompling/konvens2019/refs/heads/master/data/gold/balanced/annotations/wikipedia.conll --no-verbose -P corpora/
