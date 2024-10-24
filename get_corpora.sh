#!/bin/bash

wget https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/tiger-corpus/download/tigercorpus-2.2.conll09.tar.gz --no-verbose -P corpora/
tar -xzvf corpora/tigercorpus-2.2.conll09.tar.gz -C corpora/
