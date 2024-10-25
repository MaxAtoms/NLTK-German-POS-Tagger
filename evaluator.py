import csv
import argparse
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
        test_sentences, ignore_second_half_of_split = load_conll(test_file, percentage=percentage)
        print(test_sentences)
    else:
        try:
            folder_path = os.path.join("datasets", f"{data}")
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

    parser = argparse.ArgumentParser(description="POS Tagger Training Arguments")

    parser.add_argument("--data", type=str, default=1, help="What dataset to use")
    parser.add_argument("--testfile", type=str, help="What dataset to use")
    parser.add_argument("--model", type=int, default=1, help="Which model to use")
    parser.add_argument("--description", type=str, default="", help="Description that gets added to the result file")

    args = parser.parse_args()
    
    print(f'Loading dataset {args.data} and model {args.model}')
    
    tagger = PerceptronTagger(args.model, args.data, load=True, lang='deu')
    test_file_from_training = True if args.testfile == None else False
    shuffle = False
    
    correct, tags, accuracy = evaluate_model(tagger, args.testfile, test_file_from_training, args.data, args.model,shuffle, percentage=20)

    filename = 'results.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        fieldnames = ['name', 'accuracy', 'correct_tags', 'total_tags', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'name': args.data, 
            'accuracy': accuracy, 
            'correct_tags': correct,
            'total_tags': tags,
            'description': args.description})

    print(f"Total tags: {tags}")
    print(f"Correct tags: {correct}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
