# Create a data set with 90% of the TIGER corpus for training and 10% for the test set and train the model.
# Train on 1 iteration. (Model 1)
python german_tagger.py --train=1 --create=1 --data=tiger_tagger --iter=1

# Train on TIGER corpus for 5 iterations with 90/10 split (Model 2)
python german_tagger.py --train=1 --create=0 --data=tiger_tagger --iter=5

# Train on TIGER corpus for 10 iterations with 90/10 split (Model 3)
python german_tagger.py --train=1 --create=0 --data=tiger_tagger --iter=10

# Train on TIGER corpus for 15 iterations with 90/10 split (Model 4)
python german_tagger.py --train=1 --create=0 --data=tiger_tagger --iter=15


# -------------------------------------------------------------------------------------------- #
# create dataset with 100% of the TIGER corpus for training and train the model
python german_tagger.py --train=1 --create=1 --data=universal_tagger --iter=1 --percentage=100 # (Model 1)
python german_tagger.py --train=1 --create=0 --data=universal_tagger --iter=5 --percentage=100 # (Model 2)
python german_tagger.py --train=1 --create=0 --data=universal_tagger --iter=10 --percentage=100 # (Model 3)
python german_tagger.py --train=1 --create=0 --data=universal_tagger --iter=15 --percentage=100 # (Model 4)

# ================================================================================================= #
# ================================================================================================= #
# ================================================================================================= #

# Evaluate Model on TIGER corpus with 90/10 split with the different iterations in training:
python evaluator.py --data=tiger_tagger --model=1 --description="90/10 split. 1 iteration. Tagdict True" # 1 iteration
python evaluator.py --data=tiger_tagger --model=2 --description="90/10 split. 5 iteration. Tagdict True" # 5 iterations
python evaluator.py --data=tiger_tagger --model=3 --description="90/10 split. 10 iteration. Tagdict True" # 10 iterations
python evaluator.py --data=tiger_tagger --model=4 --description="90/10 split. 15 iteration. Tagdict True" # 15 iterations

# -------------------------------------------------------------------------------------------- #
#without tagdict
python evaluator.py --data=tiger_tagger --model=1 --tagdict=0 --description="90/10 split. 1 iteration. Tagdict False" # 1 iteration
python evaluator.py --data=tiger_tagger --model=2 --tagdict=0 --description="90/10 split. 5 iteration. Tagdict False" # 5 iterations
python evaluator.py --data=tiger_tagger --model=3 --tagdict=0 --description="90/10 split. 10 iteration. Tagdict False" # 10 iterations
python evaluator.py --data=tiger_tagger --model=4 --tagdict=0 --description="90/10 split. 15 iteration. Tagdict False" # 15 iterations


# -------------------------------------------------------------------------------------------- #

# Evaluate the model fully trained on TIGER to see if it generalizes well. Using 1 iteration
python evaluator.py --data=universal_tagger --testfile=corpora/novelette.conll --model=1 --percentage=100 --description="TIGER trained model on novelette. 1 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/ted.conll --model=1 --percentage=100 --description="TIGER trained model on ted. 1 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/sermononline.conll --model=1 --percentage=100 --description="TIGER trained model on sermononline. 1 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/wikipedia.conll --model=1 --percentage=100 --description="TIGER trained model on wikipedia. 1 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/opensubtitles.conll --model=1 --percentage=100 --description="TIGER trained model on opensubtitles. 1 iteration"

# Evaluate the model fully trained on TIGER to see if it generalizes well. Using 5 iteration
python evaluator.py --data=universal_tagger --testfile=corpora/novelette.conll --model=2 --percentage=100 --description="TIGER trained model on novelette. 5 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/ted.conll --model=2 --percentage=100 --description="TIGER trained model on ted. 5 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/sermononline.conll --model=2 --percentage=100 --description="TIGER trained model on sermononline. 5 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/wikipedia.conll --model=2 --percentage=100 --description="TIGER trained model on wikipedia. 5 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/opensubtitles.conll --model=2 --percentage=100 --description="TIGER trained model on opensubtitles. 5 iteration"

# Evaluate the model fully trained on TIGER to see if it generalizes well. Using 10 iteration
python evaluator.py --data=universal_tagger --testfile=corpora/novelette.conll --model=3 --percentage=100 --description="TIGER trained model on novelette. 10 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/ted.conll --model=3 --percentage=100 --description="TIGER trained model on ted. 10 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/sermononline.conll --model=3 --percentage=100 --description="TIGER trained model on sermononline. 10 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/wikipedia.conll --model=3 --percentage=100 --description="TIGER trained model on wikipedia. 10 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/opensubtitles.conll --model=3 --percentage=100 --description="TIGER trained model on opensubtitles. 10 iteration"

# Evaluate the model fully trained on TIGER to see if it generalizes well. Using 15 iteration
python evaluator.py --data=universal_tagger --testfile=corpora/novelette.conll --model=4 --percentage=100 --description="TIGER trained model on novelette. 15 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/ted.conll --model=4 --percentage=100 --description="TIGER trained model on ted. 15 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/sermononline.conll --model=4 --percentage=100 --description="TIGER trained model on sermononline. 15 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/wikipedia.conll --model=4 --percentage=100 --description="TIGER trained model on wikipedia. 15 iteration"
python evaluator.py --data=universal_tagger --testfile=corpora/opensubtitles.conll --model=4 --percentage=100 --description="TIGER trained model on opensubtitles. 15 iteration"
# -------------------------------------------------------------------------------------------- #
# compare to classifier approach with 90/10 split:
cd comparison
python train_eval.py


