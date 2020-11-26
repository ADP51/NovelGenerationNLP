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


def arthur_conan_doyle():
    test = WordModel('arthur-conan-doyle', 'E:/NovelGenerationNLP/test_models/',
                     ['holmes', 'watson', 'gun', 'war', 'mystery', 'murder', 'woman'])

    test.w2v_grams(author='Arthur Conan Doyle', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()
    test.gen_train(epochs=80, batch_size=4096, rnn_units=128)
    test.w2v_seeds(['Elementary, my dear Watson.', 'When you have eliminated all which is impossible,',
                    'There is nothing more deceptive than an obvious fact.', 'You see, but you do not observe.',
                    'Take a community of Dutchmen of the type of those who defended themselves',
                    'My name is Sherlock Holmes.'
                    ], log=False)

    return test

def mark_twain():
    test = WordModel('mark-twain', 'E:/NovelGenerationNLP/test_models/',
                     ['sawyer', 'science', 'story', 'war', 'mississippi', 'america', 'woman'])

    test.w2v_grams(author='Mark Twain', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()
    test.gen_train(epochs=80, batch_size=4096, rnn_units=128)
    test.w2v_seeds(['It is a time when one’s spirit is subdued and sad',
                    'If you should rear a duck in the heart of the Sahara',
                    'Now and then we had a hope that if we lived and were good',
                    'The Mississippi River towns are comely, clean, well built, and pleasing to the eye',
                    'All right, then, I\'ll go to hell',
                    'If you tell the truth you do not need a good memory'
                    ], log=False)

    return test

def simpsons():
    test = WordModel('simpsons', 'E:/NovelGenerationNLP/test_models/',
                     ['homer', 'donut', 'duff', 'kill', 'mayor', 'prank', 'springfield'])

    test.w2v_grams(author='Matt Groening', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()
    test.gen_train(epochs=80, batch_size=4096, rnn_units=128)
    test.w2v_seeds(['We were gonna keep the gray one, but the mother ate her',
                    'You see, class, my Lyme Disease turned out to be psychosomatic',
                    'Alright, alright, spilled milk, spilled milk, spilled milk...',
                    'Would you like something to eat? I\'ve got dried apricots...',
                    'Wait a minute, this unkempt youngster just might be on to something',
                    'Earth base, this is commander Bart McCool. We are under attack by the Zorrinid Brain Changers!'
                    ], log=False)

    return test

def william_shakespeare():
    test = WordModel('william-shakespeare', 'E:/NovelGenerationNLP/test_models/',
                     ['romeo', 'thumb', 'wicked', 'world', 'love', 'beware', 'havoc'])

    test.w2v_grams(author='William Shakespeare', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()
    test.gen_train(epochs=80, batch_size=4096, rnn_units=128)
    test.w2v_seeds(['All that glitters is not gold',
                    'Hell is empty and all the devils are here.',
                    'By the pricking of my thumbs, Something wicked this way comes.',
                    'The lady doth protest too much, methinks.',
                    'What’s in a name? A rose by any other name would smell as sweet.',
                    'Friends, Romans, countrymen, lend me your ears'
                    ], log=False)

    return test

def edgar_allan_poe():
    test = WordModel('edgar-allan-poe', 'E:/NovelGenerationNLP/test_models/',
                     ['raven', 'heart', 'death', 'insane', 'love', 'sea', 'beauty'])

    test.w2v_grams(author='Edgar Allan Poe', sentence_len=8, sentence_offset=1)
    test.w2v_grams_to_file()
    test.w2v_train(workers=6)
    test.w2v_model_to_file()
    test.gen_train(epochs=80, batch_size=4096, rnn_units=128)
    test.w2v_seeds(['Those who dream by day are cognizant of many things',
                    'I have great faith in fools - self-confidence my friends will call it',
                    'Once upon a midnight dreary, while I pondered, weak and weary,',
                    'Sleep, those little slices of death — how I loathe them.',
                    'It was many and many a year ago, In a kingdom by the sea,',
                    'The true genius shudders at incompleteness'
                    ], log=False)

    # test.w2v_model_to_file()
    # test.w2v_grams_to_file()

    return test


if __name__ == '__main__':
    test = None
