from german_tagger import PerceptronTagger, load_conll
from nltk.data import path
from tqdm import tqdm 
import os
import random

def load_test_sentences_from_file(file_path, shuffle=True):
    test_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = [tuple(word_tag.split('°')) for word_tag in line.strip().split()]
            test_sentences.append(sentence)
    if shuffle:
        random.shuffle(test_sentences)
    return test_sentences

def evaluate_model(tagger, test_file, test_file_from_training, data, model, shuffle, percentage=100):
    if(not test_file_from_training):
        test_sentences = load_conll(test_file, percentage=percentage)
    else:
        try:
            folder_path = os.path.join("datasets", f"data{data}")
            test_file_path = os.path.join(folder_path, "test_sentences.txt")
            test_sentences = load_test_sentences_from_file(test_file_path, shuffle)
        except FileNotFoundError:
            print("No training file found")
            exit(1)
    
    total_correct = 0
    total_tags = 0
    with tqdm(total=len(test_sentences)) as pbar:
        for sentence in test_sentences:
            words, true_tags = zip(*sentence)
            
            predicted_tags = [tag for word, tag in tagger.tag(words)]
            #for word, tag, pred_tag in zip(words, true_tags, predicted_tags):
            #    if tag != pred_tag:
            #        print(f"Word: {word}, True Tag: {tag}, Predicted Tag: {pred_tag}")
            total_correct += sum(1 for true_tag, pred_tag in zip(true_tags, predicted_tags) if true_tag == pred_tag)
            total_tags += len(true_tags)
            pbar.update(1)

    accuracy = total_correct / total_tags if total_tags > 0 else 0
    return total_correct, total_tags, accuracy

if __name__ == "__main__":
    path.append(os.getcwd())
    
    #which model to load?
    data = 2
    model = 1
    
    tagger = PerceptronTagger(model, data, load=True, lang='deu')
    test_file = "tiger_release_aug07.corrected.16012013.Conll09" 
    test_file_from_training = True
    shuffle = False
    
    correct, tags, accuracy = evaluate_model(tagger, test_file, test_file_from_training, data, model,shuffle, percentage=20)
    print(f"Total tags: {tags}")
    print(f"Correct tags: {correct}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
