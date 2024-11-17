import argparse
from collections import defaultdict

def process_conll_file(filename):
    num_sentences = 0
    num_tokens = 0
    word_pos_map = defaultdict(set)

    with open(filename, 'r', encoding='utf-8') as file:
        sentence_started = False
        for line in file:
            line = line.strip()
            if line == "":  # Sentence boundary
                if sentence_started:
                    num_sentences += 1
                    sentence_started = False
                continue

            sentence_started = True
            columns = line.split()

            if len(columns) < 4:
                raise ValueError("CoNLL file is not properly formatted.")

            word, pos_tag = columns[1], columns[4]
            num_tokens += 1
            word_pos_map[word].add(pos_tag)

        # Account for the last sentence if the file does not end with a blank line
        if sentence_started:
            num_sentences += 1

    num_types = len(word_pos_map)
    unambiguous_types = sum(1 for tags in word_pos_map.values() if len(tags) == 1)

    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of types: {num_types}")
    print(f"Number of unambiguous types: {unambiguous_types}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CoNLL file.")
    parser.add_argument("filename", type=str, help="Path to the CoNLL09 file")
    args = parser.parse_args()
    process_conll_file(args.filename)

