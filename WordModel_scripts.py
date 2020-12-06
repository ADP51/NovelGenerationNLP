from keras.models import model_from_json

from WordModel import WordModel


def doyle_test1():
    test = WordModel('doyle_test1', 'E:/NovelGenerationNLP/test_models/',
                     ['holmes', 'watson', 'gun', 'war', 'mystery', 'murder', 'woman'])

    test.w2v_grams(author='Arthur Conan Doyle', sentence_len=8, sentence_offset=1)
    test.w2v_train(workers=6)
    # test.w2v_grams_to_file()
    # test.w2v_model_to_file()

    # test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/doyle_test1_grams.txt')
    # test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/doyle_test1_model.model')

    test.gen_train(epochs=1, batch_size=4096, rnn_units=128)
    # test.w2v_seeds(['Elementary, my dear Watson.', 'When you have eliminated all which is impossible,',
    #                 'There is nothing more deceptive than an obvious fact.', 'You see, but you do not observe.',
    #                 'Take a community of Dutchmen of the type of those who defended themselves',
    #                 'My name is Sherlock Holmes.'
    #                 ], log=False)

    return test


def arthur_conan_doyle_w2v():
    test = WordModel('arthur-conan-doyle', 'E:/NovelGenerationNLP/test_models/',
                     ['holmes', 'watson', 'gun', 'war', 'mystery', 'murder', 'woman'])

    test.w2v_grams(author='Arthur Conan Doyle', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['Elementary, my dear Watson.',
                    'When you have eliminated all which is impossible,',
                    'There is nothing more deceptive than an obvious fact.',
                    'You see, but you do not observe.',
                    'Take a community of Dutchmen of the type of those who defended themselves',
                    'My name is Sherlock Holmes.'
                    ], log=False)

    return test


def arthur_conan_doyle_gen(epochs: int = 40):
    test = WordModel('arthur-conan-doyle', 'E:/NovelGenerationNLP/test_models/',
                     ['holmes', 'watson', 'gun', 'war', 'mystery', 'murder', 'woman'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/arthur-conan-doyle_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/arthur-conan-doyle_model.model')

    test.gen_train(epochs=epochs, batch_size=4096, rnn_units=64)
    return test


def arthur_conan_doyle_output(seed: str):

    test = WordModel('arthur-conan-doyle', 'E:/NovelGenerationNLP/test_models/',
                     ['holmes', 'watson', 'gun', 'war', 'mystery', 'murder', 'woman'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/arthur-conan-doyle_model.model')

    with open('E:/NovelGenerationNLP/test_models/arthur-conan-doyle_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/arthur-conan-doyle_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test


def mark_twain_w2v():
    test = WordModel('mark-twain', 'E:/NovelGenerationNLP/test_models/',
                     ['sawyer', 'science', 'story', 'war', 'mississippi', 'america', 'woman'])

    test.w2v_grams(author='Mark Twain', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['It is a time when one’s spirit is subdued and sad',
                    'If you should rear a duck in the heart of the Sahara',
                    'Now and then we had a hope that if we lived and were good',
                    'The Mississippi River towns are comely, clean, well built, and pleasing to the eye',
                    'All right, then, I\'ll go to hell',
                    'If you tell the truth you do not need a good memory'
                    ], log=False)

    return test


def mark_twain_gen(epochs: int = 40):
    test = WordModel('mark-twain', 'E:/NovelGenerationNLP/test_models/',
                     ['sawyer', 'science', 'story', 'war', 'mississippi', 'america', 'woman'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/mark-twain_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/mark-twain_model.model')

    test.gen_train(epochs=epochs, batch_size=4096, rnn_units=64)

    return test


def mark_twain_output(seed: str):

    test = WordModel('mark-twain', 'E:/NovelGenerationNLP/test_models/',
                     ['sawyer', 'science', 'story', 'war', 'mississippi', 'america', 'woman'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/mark-twain_model.model')

    with open('E:/NovelGenerationNLP/test_models/mark-twain_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/mark-twain_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test


def simpsons_w2v():
    test = WordModel('simpsons', 'E:/NovelGenerationNLP/test_models/',
                     ['homer', 'donut', 'duff', 'kill', 'mayor', 'prank', 'springfield'])

    test.w2v_grams(author='Matt Groening', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['We were gonna keep the gray one, but the mother ate her',
                    'I didn\'t vote. Voting\'s for geeks.',
                    'Alright, alright, spilled milk, spilled milk, spilled milk...',
                    'Would you like something to eat? I\'ve got dried apricots...',
                    'Wait a minute, this unkempt youngster just might be on to something',
                    'Earth base, this is commander Bart McCool. We are under attack by the Zorrinid Brain Changers!'
                    ], log=False)

    return test


def simpsons_gen(epochs: int = 40):
    test = WordModel('simpsons', 'E:/NovelGenerationNLP/test_models/',
                     ['homer', 'donut', 'duff', 'kill', 'mayor', 'prank', 'springfield'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/simpsons_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/simpsons_model.model')

    test.gen_train(epochs=epochs, batch_size=4096, rnn_units=64)

    return test

def simpsons_output(seed: str):
    test = WordModel('simpsons', 'E:/NovelGenerationNLP/test_models/',
                     ['homer', 'donut', 'duff', 'kill', 'mayor', 'prank', 'springfield'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/simpsons_model.model')

    with open('E:/NovelGenerationNLP/test_models/simpsons_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/simpsons_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test

def william_shakespeare_w2v():
    test = WordModel('william-shakespeare', 'E:/NovelGenerationNLP/test_models/',
                     ['romeo', 'thumb', 'wicked', 'world', 'love', 'beware', 'havoc'])

    test.w2v_grams(author='William Shakespeare', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['All that glitters is not gold',
                    'Hell is empty and all the devils are here.',
                    'By the pricking of my thumbs, Something wicked this way comes.',
                    'The lady doth protest too much, methinks.',
                    'What’s in a name? A rose by any other name would smell as sweet.',
                    'Friends, Romans, countrymen, lend me your ears'
                    ], log=False)

    return test


def william_shakespeare_gen(epochs: int = 40):
    test = WordModel('william-shakespeare', 'E:/NovelGenerationNLP/test_models/',
                     ['romeo', 'thumb', 'wicked', 'world', 'love', 'beware', 'havoc'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/william-shakespeare_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/william-shakespeare_model.model')

    test.gen_train(epochs=epochs, batch_size=4096, rnn_units=64)

    return test


def william_shakespeare_output(seed: str):

    test = WordModel('william-shakespeare', 'E:/NovelGenerationNLP/test_models/',
                     ['romeo', 'thumb', 'wicked', 'world', 'love', 'beware', 'havoc'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/william-shakespeare_model.model')

    with open('E:/NovelGenerationNLP/test_models/william-shakespeare_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/william-shakespeare_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test


def edgar_allan_poe_w2v():
    test = WordModel('edgar-allan-poe', 'E:/NovelGenerationNLP/test_models/',
                     ['raven', 'heart', 'death', 'insane', 'love', 'sea', 'beauty'])

    test.w2v_grams(author='Edgar Allan Poe', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['Those who dream by day are cognizant of many things',
                    'I have great faith in fools - self-confidence my friends will call it',
                    'Once upon a midnight dreary, while I pondered, weak and weary,',
                    'Sleep, those little slices of death — how I loathe them.',
                    'It was many and many a year ago, In a kingdom by the sea,',
                    'I became insane, with long intervals of horrible sanity.'
                    ], log=False)

    return test


def edgar_allan_poe_gen(epochs: int = 40):
    test = WordModel('edgar-allan-poe', 'E:/NovelGenerationNLP/test_models/',
                     ['raven', 'heart', 'death', 'insane', 'love', 'sea', 'beauty'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/edgar-allan-poe_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/edgar-allan-poe_model.model')

    test.gen_train(epochs=epochs, batch_size=4096, rnn_units=64)

    return test


def edgar_allan_poe_output(seed: str):

    test = WordModel('edgar-allan-poe', 'E:/NovelGenerationNLP/test_models/',
                     ['raven', 'heart', 'death', 'insane', 'love', 'sea', 'beauty'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/edgar-allan-poe_model.model')

    with open('E:/NovelGenerationNLP/test_models/edgar-allan-poe_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/edgar-allan-poe_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test


def amanda_mckittrick_ros_w2v():
    test = WordModel('amanda-mckittrick-ros', 'E:/NovelGenerationNLP/test_models/',
                     ['glory', 'fortune', 'house', 'summer', 'love', 'my', 'beauty'])

    test.w2v_grams(author='Amanda McKittrick Ros', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()

    test.w2v_seeds(['When on the eve of glory, whilst brooding over the prospects of a bright and happy future,',
                    'Sympathise with me, indeed! Ah, no!',
                    'The December sun had hidden its dull rays behind the huge rocks that rose monstrously high',
                    'Arouse the seeming deadly creature to that standard of joy and gladness which should mark his noble path!',
                    'The silvery touch of fortune is too often gilt with betrayal',
                    'Our hopes when elevated to that standard of ambition which demands unison may fall asunder like an ancient ruin.'
                    ], log=False)

    return test


def amanda_mckittrick_ros_gen(epochs: int = 40):
    test = WordModel('amanda-mckittrick-ros', 'E:/NovelGenerationNLP/test_models/',
                     ['glory', 'fortune', 'house', 'summer', 'love', 'my', 'beauty'])

    test.w2v_grams_from_file('E:/NovelGenerationNLP/test_models/amanda-mckittrick-ros_grams.txt')
    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/amanda-mckittrick-ros_model.model')

    test.gen_train(epochs=epochs, batch_size=1024, rnn_units=64)

    return test


def amanda_mckittrick_ros_output(seed: str):

    test = WordModel('amanda-mckittrick-ros', 'E:/NovelGenerationNLP/test_models/',
                     ['glory', 'fortune', 'house', 'summer', 'love', 'my', 'beauty'])

    test.w2v_model_from_file('E:/NovelGenerationNLP/test_models/amanda-mckittrick-ros_model.model')

    with open('E:/NovelGenerationNLP/test_models/amanda-mckittrick-ros_model.json', 'r') as file:
        json_config = file.read()

    test.model = model_from_json(json_config)

    print(test.model.summary())

    test.model.load_weights("E:/NovelGenerationNLP/test_models/amanda-mckittrick-ros_model.ckpt")
    #
    # print(test.model.summary())

    print(test._gen_generate_next(seed, num_generated=100))
    # print(test._gen_generate_next("My name is William Shakespeare"))

    return test


if __name__ == '__main__':
    test = None
