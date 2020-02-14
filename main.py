import random

import spacy

PROPN = []
NUM = []
NOUN = []
VERB = []
PUNCT = []
ADP = []
SYM = []
PRON = []
DET = []
ADJ = []
PART = []
ADV = []
INTJ = []
SENT_STRUCTS = []

nlp = spacy.load("en_core_web_sm")


def main():
    # reading in a text file for to model
    with open("./data/someText.txt", "r") as myfile:
        data = myfile.read()

    myfile.close()

    # remove newlines from text
    data.rstrip('\n')

    # split into list of sentences
    text = data.split('.')

    # for each sentence in text create sentence mapping and add words to bags
    for sent in text:
        sent.strip()
        doc = nlp(sent)
        create_sent_map(doc)

    generate_random_sent(20)


# create sentence strucures and add new words to their respective lists
def create_sent_map(sent):
    map = ""
    for token in sent:
        map = map + " " + token.pos_
        if token.pos_ in "NOUN" and token.text not in NOUN:
            NOUN.append(token.text.lower())
        elif token.pos_ in "VERB" and token.text not in VERB:
            VERB.append(token.text.lower())
        elif token.pos_ in "PROPN" and token.text not in PROPN:
            PROPN.append(token.text)
        elif token.pos_ in "NUM" and token.text not in NUM:
            NUM.append(token.text.lower())
        elif token.pos_ in "PUNCT" and token.text not in PUNCT:
            PUNCT.append(token.text.lower())
        elif token.pos_ in "ADP" and token.text not in ADP:
            ADP.append(token.text.lower())
        elif token.pos_ in "SYM" and token.text not in SYM:
            SYM.append(token.text.lower())
        elif token.pos_ in "PRON" and token.text not in PRON:
            PRON.append(token.text.lower())
        elif token.pos_ in "DET" and token.text not in DET:
            DET.append(token.text.lower())
        elif token.pos_ in "ADJ" and token.text not in ADJ:
            ADJ.append(token.text.lower())
        elif token.pos_ in "PART" and token.text not in PART:
            PART.append(token.text.lower())
        elif token.pos_ in "ADV" and token.text not in ADV:
            ADV.append(token.text.lower())
        elif token.pos_ in "INTJ" and token.text not in INTJ:
            INTJ.append(token.text.lower())
    if map not in SENT_STRUCTS:
        SENT_STRUCTS.append(map)


# generate a given number of random sentences
def generate_random_sent(num):
    for x in range(num):
        gen_sentence = ""
        struct = random.choice(SENT_STRUCTS)
        split_struct = struct.split()

        for word in split_struct:
            if word in "NOUN":
                gen_sentence = gen_sentence + " " + random.choice(NOUN)
            elif word in "VERB":
                gen_sentence = gen_sentence + " " + random.choice(VERB)
            elif word in "PROPN":
                gen_sentence = gen_sentence + " " + random.choice(PROPN)
            elif word in "NUM":
                gen_sentence = gen_sentence + " " + random.choice(NUM)
            elif word in "ADP":
                gen_sentence = gen_sentence + " " + random.choice(ADP)
            elif word in "SYM":
                gen_sentence = gen_sentence + " " + random.choice(SYM)
            elif word in "PRON":
                gen_sentence = gen_sentence + " " + random.choice(PRON)
            elif word in "DET":
                gen_sentence = gen_sentence + " " + random.choice(DET)
            elif word in "ADJ":
                gen_sentence = gen_sentence + " " + random.choice(ADJ)
            elif word in "PART":
                gen_sentence = gen_sentence + " " + random.choice(PART)
            elif word in "ADV":
                gen_sentence = gen_sentence + " " + random.choice(ADV)
            elif word in "INTJ":
                gen_sentence = gen_sentence + " " + random.choice(INTJ)
        print(gen_sentence)
    print()


main()



