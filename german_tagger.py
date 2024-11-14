# This module is a port of the Textblob Averaged Perceptron Tagger
# Author: Matthew Honnibal <honnibal+gh@gmail.com>,
#         Long Duong <longdt219@gmail.com> (NLTK port)
# URL: <https://github.com/sloria/textblob-aptagger>
#      <https://www.nltk.org/>
# Copyright 2013 Matthew Honnibal
# NLTK modifications Copyright 2015 The NLTK Project
#
# This module is provided under the terms of the MIT License.

import os
import argparse
import json
import logging
import random
from collections import defaultdict
from tqdm import tqdm 
from nltk import jsontags
from nltk.data import find, load, path
from nltk.tag.api import TaggerI
import os
try:
    import numpy as np
except ImportError:
    pass

TRAINED_TAGGER_PATH = "taggers/"

TAGGER_JSONS = {
    "eng": {
        "weights": "averaged_perceptron_tagger_eng.weights.json",
        "tagdict": "averaged_perceptron_tagger_eng.tagdict.json",
        "classes": "averaged_perceptron_tagger_eng.classes.json",
    },
    "rus": {
        "weights": "averaged_perceptron_tagger_rus.weights.json",
        "tagdict": "averaged_perceptron_tagger_rus.tagdict.json",
        "classes": "averaged_perceptron_tagger_rus.classes.json",
    },
    "deu": {
        "weights": "averaged_perceptron_tagger_deu.weights.json",
        "tagdict": "averaged_perceptron_tagger_deu.tagdict.json",
        "classes": "averaged_perceptron_tagger_deu.classes.json",
    },
}


@jsontags.register_tag
class AveragedPerceptron:
    """An averaged perceptron, as implemented by Matthew Honnibal.

    See more implementation details here:
        https://explosion.ai/blog/part-of-speech-pos-tagger-in-python
    """

    json_tag = "nltk.tag.perceptron.AveragedPerceptron"

    def __init__(self, weights=None):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = weights if weights else {}
        #self.classes = set()
        self.classes = set()
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def _softmax(self, scores):
        s = np.fromiter(scores.values(), dtype=float)
        exps = np.exp(s)
        return exps / np.sum(exps)

    def predict(self, features, return_conf=False):#, clas=None):
        """Dot-product the features and current weights and return the best label."""
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
        # Do a secondary alphabetic sort, for stability
        best_label = max(self.classes, key=lambda label: (scores[label], label))
        # compute the confidence
        conf = max(self._softmax(scores)) if return_conf == True else None

        return best_label, conf

    def update(self, truth, guess, features):
        """Update the feature weights."""

        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)

    def average_weights(self):
        """Average weights from all iterations."""
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / self.i, 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights

    def save(self, path):
        """Save the model weights as json"""
        with open(path, "w") as fout:
            return json.dump(self.weights, fout)

    def load(self, path):
        """Load the json model weights."""
        with open(path) as fin:
            self.weights = json.load(fin)

    def encode_json_obj(self):
        return self.weights

    @classmethod
    def decode_json_obj(cls, obj):
        return cls(obj)


@jsontags.register_tag
class PerceptronTagger(TaggerI):
    """
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
    https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

    >>> from nltk.tag.perceptron import PerceptronTagger

    Train the model

    >>> tagger = PerceptronTagger(load=False)

    >>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
    ... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

    >>> tagger.tag(['today','is','a','beautiful','day'])
    [('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]

    Use the pretrain model (the default constructor)

    >>> pretrain = PerceptronTagger()

    >>> pretrain.tag('The quick brown fox jumps over the lazy dog'.split())
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

    >>> pretrain.tag("The red cat".split())
    [('The', 'DT'), ('red', 'JJ'), ('cat', 'NN')]
    """

    json_tag = "nltk.tag.sequential.PerceptronTagger"

    START = ["-START-", "-START2-"]
    END = ["-END-", "-END2-"]

    def __init__(self, model=-1, dataset=-1, load=True, lang='deu'): #eng
        """
        :param load: Load the json model upon instantiation.
        """
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load_from_json(model, dataset, lang)


    def tag(self, tokens, return_conf=False, use_tagdict=True):
        """
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        """
        prev, prev2 = self.START
        output = []

        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag, conf = (
                (self.tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
            )
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, conf = self.model.predict(features, return_conf)
            output.append((word, tag, conf) if return_conf == True else (word, tag))

            prev2 = prev
            prev = tag

        return output

    def train(self, sentences, data, model_number, save_loc=None, nr_iter=5):
        """Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a json model in this location.
        :param nr_iter: Number of training iterations.
        """
        # We'd like to allow ``sentences`` to be either a list or an iterator,
        # the latter being especially important for a large training dataset.
        # Because ``self._make_tagdict(sentences)`` runs regardless, we make
        # it populate ``self._sentences`` (a list) with all the sentences.
        # This saves the overheard of just iterating through ``sentences`` to
        # get the list by ``sentences = list(sentences)``.

        self._sentences = list()  # to be populated by self._make_tagdict...
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            with tqdm(total=len(self._sentences), desc=f"Iteration {iter_ + 1}/{nr_iter}") as pbar:
                for sentence in self._sentences:
                    words, tags = zip(*sentence)

                    prev, prev2 = self.START
                    context = self.START + [self.normalize(w) for w in words] + self.END
                    for i, word in enumerate(words):
                        guess = self.tagdict.get(word)
                        if not guess:
                            feats = self._get_features(i, word, context, prev, prev2)
                            guess, _ = self.model.predict(feats)
                            self.model.update(tags[i], guess, feats)
                        prev2 = prev
                        prev = guess
                        c += guess == tags[i]
                        n += 1
                    pbar.update(1)
            random.shuffle(self._sentences)
            logging.info(f"Iter {iter_}: {c}/{n}={_pc(c, n)}")

        # We don't need the training sentences anymore, and we don't want to
        # waste space on them when we the trained tagger.
        self._sentences = None

        self.model.average_weights()
        # Save to json files.
        if save_loc is not None:
            self.save_to_json(save_loc, data, model_number)
    
    def save_to_json(self, loc, data, model_number, lang="deu"):
        base_dir = os.path.join(loc, f"averaged_perceptron_tagger_{lang}/{data}")
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        model_dir = os.path.join(base_dir, f"model{model_number}")
        os.makedirs(model_dir)
    
        # Save the JSON files in the model directory
        with open(os.path.join(model_dir, TAGGER_JSONS[lang]["weights"]), "w") as fout:
            json.dump(self.model.weights, fout)
        with open(os.path.join(model_dir, TAGGER_JSONS[lang]["tagdict"]), "w") as fout:
            json.dump(self.tagdict, fout)

        with open(os.path.join(model_dir, TAGGER_JSONS[lang]["classes"]), "w") as fout:
            json.dump(list(self.classes), fout)

        print(f"Model saved in {model_dir}")


    def load_from_json(self, model, dataset, lang="deu"):
        loc = find(f"taggers/averaged_perceptron_tagger_{lang}/{dataset}/model{model}/")

        with open(loc + TAGGER_JSONS[lang]["weights"]) as fin:
            print("Loading weights...")
            self.model.weights = json.load(fin)
            print("Weights loaded successfully.")

        with open(loc + TAGGER_JSONS[lang]["tagdict"]) as fin:
            print("Loading tag dictionary...")
            self.tagdict = json.load(fin)
            print("Tag dictionary loaded successfully.")

        with open(loc + TAGGER_JSONS[lang]["classes"]) as fin:
            print("Loading class...")
            data = json.load(fin)
            self.classes = set(data)
            print("Classes loaded successfully:")
        self.model.classes = self.classes

    def encode_json_obj(self):
        return self.model.weights, self.tagdict, list(self.classes)

    @classmethod
    def decode_json_obj(cls, obj):
        tagger = cls(load=False)
        tagger.model.weights, tagger.tagdict, tagger.classes = obj
        tagger.classes = set(tagger.classes)
        tagger.model.classes = tagger.classes
        return tagger

    def normalize(self, word):
        """
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        """
        if "-" in word and word[0] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word and word[0].isdigit():
            return "!DIGITS"
        return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        """

        def add(name, *args):
            features[" ".join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add("bias")
        add("i suffix", word[-3:])
        add("i pref1", word[0] if word else "")
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1])
        add("i-1 suffix", context[i - 1][-3:])
        add("i-2 word", context[i - 2])
        add("i+1 word", context[i + 1])
        add("i+1 suffix", context[i + 1][-3:])
        add("i+2 word", context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        """
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        """
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            self._sentences.append(sentence)
            for word, tag in sentence:
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (mode / n) >= ambiguity_thresh:
                self.tagdict[word] = tag


def _pc(n, d):
    return (n / d) * 100


# no load conell from github
# no get pretrained model

def load_train_sentences_from_file(file_path):
    train_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = [tuple(word_tag.split('°')) for word_tag in line.strip().split()]
            train_sentences.append(sentence)
    return train_sentences

def train_perceptron_tagger_german(train_file, data, model_number, train_model=True, save_loc=None, nr_iter=5, percentage=100, createfiles=False):

    if createfiles:
        train_sentences, test_sentences = load_conll(train_file, percentage)
        
        if not os.path.exists("datasets"):
            os.makedirs("datasets")

        folder_path = os.path.join("datasets", data)
        os.makedirs(folder_path)


        train_file_path = os.path.join(folder_path, "train_sentences.txt")
        test_file_path = os.path.join(folder_path, "test_sentences.txt")

        #train
        with open(train_file_path, 'w', encoding='utf-8') as train_file:
            for sentence in train_sentences:
                train_file.write(' '.join([f"{word}°{tag}" for word, tag in sentence]) + '\n')

        #test
        with open(test_file_path, 'w', encoding='utf-8') as test_file:
            for sentence in test_sentences:
                test_file.write(' '.join([f"{word}°{tag}" for word, tag in sentence]) + '\n')

        print(f"Training sentences written to {train_file_path}")
        print(f"Testing sentences written to {test_file_path}")
    else:
        folder_path = os.path.join("datasets", f"{data}")
        train_file_path = os.path.join(folder_path, "train_sentences.txt")
        train_sentences = load_train_sentences_from_file(train_file_path)
        random.shuffle(train_sentences)
        
    if train_model:
        print("Training the German Perceptron Tagger...")
        
        # Train the tagger
        tagger = PerceptronTagger(load=False, lang='deu')

        tagger.train(train_sentences, data, model_number, save_loc=save_loc, nr_iter=nr_iter)

        return tagger
    else:
        exit(0)

def load_conll(file_path, percentage=100):
    """Load the CoNLL-09 format data from the given file."""
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split('\t')
                word = parts[1] 
                tag = parts[4]  
                sentence.append((word, tag))
        if sentence:
            sentences.append(sentence)

    random.shuffle(sentences)
    
    subset_size = int(len(sentences) * (percentage / 100))

    return sentences[:subset_size], sentences[subset_size:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POS Tagger Training Arguments")

    parser.add_argument("--train", type=int, choices=[0, 1], default=1, help="Execute training (1 for True, 0 for False)")
    parser.add_argument("--create", type=int, choices=[0, 1], default=1, help="Create dataset (1 for True, 0 for False)")
    parser.add_argument("--data", type=str, required=True, help="Name")
    parser.add_argument("--dataset", type=str, default="corpora/tiger_release_aug07.corrected.16012013.conll09", help="Dataset filename")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations")
    parser.add_argument("--percentage", type=int, default=90, help="Percentage of the dataset used for training")
    parser.add_argument("--model", type=int, help="Number of the model to train")

    args = parser.parse_args()

    # _get_pretrain_model()
    path.append(os.getcwd())

    save_loc = TRAINED_TAGGER_PATH

    print(f'Training dataset "{args.data}" for {args.iter} iteration(s) using {args.percentage}% of the dataset')

    german_tagger = train_perceptron_tagger_german(train_file=args.dataset, data=args.data, train_model=args.train, model_number=args.model, save_loc=save_loc, nr_iter=args.iter, percentage=args.percentage, createfiles=args.create)    
    print("Training completed and model saved.")
