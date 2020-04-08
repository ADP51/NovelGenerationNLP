# Novel Generation

### Setting up environment
Using conda: conda env remove --name env

### Conda Cheat sheet
https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

## How it works
Zady is a machine learning program that takes in a given author's work in text format and tries to emulate them. It does this using a recursive neural network and supervised learning. Zady is built using tensorflow which is a library that makes creating neural networks and machine learning in general very accessible with it's relatively simple API. Using tensorflow we are able to specify the number of layers and the types of cells we want the neural network to use.

### Pre-Processing

When Zady ingests the text it scans through the entire corpus and finds each unique character, creating a finite vocabulary. Once it has its vocabulary it knows that when it tries to predict what the next character it should write has to be one of the ones in it's vocabulary. Once we have that vocabulary we need to create a map with integer values for each unique character in the vocabulary. We do this because the neural network needs to make predictions based on math, so it's not actually predicting the next character but rather the integer value we assign to that character. Once the rnn has finished generating it will return a bunch of integer values which we can then map back to characters.

 ### Creating training data

So once we have our text preprocessed and ready to be fed into the network, we first need to create our training data. Zady is trained using supervised learning meaning we need to make a prediction about the next character in a sequence and try to guide it in the right direction based on what the actual next character in the sequence was, basically a complicated game of hot and cold. So we take our converted text and break it into *N* character segments and create an input segment that has one less character. We then feed the input segment into the network and make a prediction then adjust based on what the original segment has as the correct character.

### Training 

Once the training data is finished and ready to be used, we build a model consisting of three layer types, an embedding layer, GRU layer, and finally a Dense layer. The model is then trained against all of the training data, after each time the training data is completely run through the model will save itself in a binary file which we can then load the trained model from.
