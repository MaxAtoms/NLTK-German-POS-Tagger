import random
from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

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

train_sents, test_sents = load_conll('../corpora/tiger_release_aug07.corrected.16012013.conll09', 90)

tagger = ClassifierBasedGermanTagger(train=train_sents)
accuracy = tagger.evaluate(test_sents)
print(accuracy)
