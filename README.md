# Novel Generation

## How It Works
Zady is a machine learning program that takes in a given author's work in text format and tries to emulate it. It does this using Word2Vec and Recurrent Neural Network algorithms. The Word2Vec process, provided by Gensim, indexes words according to their context in a corpus, and creates an initial set of contextual weights. Those weights are given to a Recurrent Neural Network, provided by TensorFlow, which trains it to predict word sequences.

## Setup

### Setting Up an Environment
The environment manager that has been used for this project is Anaconda. Some environment files have already been created in the "env" folder. The files produced by Anaconda provide instructions for creating a local instance of that environment. Otherwise, make sure to install appropriate versions of all packages used in the clean_text.py, WordModel.py, and corpus.py files, at least.

### Conda Cheat-Sheet
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

### Code Management
Any Python code manager should work, but the bulk of the work on this project was done with PyCharm and Visual Studio Code.

## Corpus and Text Cleaning

### Text Sources
The text used for this project has been gathered from free online sources such as Kaggle and Project Gutenberg.

### Text Cleaning
The general goal of text cleaning in this case is to remove all extraneous information from a source, so that only the usable content is left. Various manual adjustments, formatting changes, and cuts will be required for any text source. However, a script ("clean_text.py", under the "data" folder) with a simple user interface has been developed to remove HTML formatting and many other artefacts found in Gutenberg sources.

### Corpus
A corpus is a collection of text used to train a particular model. A class ("corpus.py") and a json file ("corpus_directory.json", under the "data" folder) has been developed to manage many corpuses used in many models. The "corpus_directory.json" file provides information about all included text sources, which are stored under the "data/corpus" folder. The "corpus.py" class provides a simple API for accessing the directory file and gathering text.

## The Word Model

### Word2Vec
Word2Vec is an NLP process that allows the development of word vectors - contextual associations of various words within a corpus. This process is used to create an indexed vocabulary of words, and to create a set of initial training weights. In the WordModel class ("WordModel.py"), methods with "w2v" in their name are associated with this process.

### RNN
A Recurrent Neural Network is a machine learning process especially suitable for NLP. It is used in this project to develop and train a model capable of generating text based on an initial seed. In the WordModel class, methods with "gen" in their name are associated with this process.

### Usage
The Word Model is represented by the WordModel class. The typical usage of this class and its methods can be seen in the script "WordModel_scripts.py". Typical usage of that script is an author's Word2Vec method ("mark_twain_w2v()", for example), followed by that author's RNN method ("mark_twain_gen()", for example). The Word2Vec or "w2v" methods produce a binary Word2Vec model file, a pickled text file containing the model's formatted corpus, and text file containing some sample seed strings. The RNN or "gen" methods produce an RNN model architecture json file, and a binary checkpoint file (or set of files) which contain the RNN model weights. All of these files can be used by the "output" methods ("mark_twain_output()", for example), or by the other programs that make up the Novel Generation project.
