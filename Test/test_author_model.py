import unittest
from author_model import build_model, generate_text, train_model, save_char_mapping
from main import open_file, read_file, load_model, compile_model
import csv


class MyTestCase(unittest.TestCase):

    # TODO the assert statement is incorrect
    def test_save_char_mapping(self):

        text = open_file()
        vocab = read_file(text)

        save_char_mapping(vocab, 'test.csv')
        self.assertEqual(csv.reader('test.csv'), csv.reader('shakespeare.csv'))

    # Tests the build a model method from author_model
    def test_build_model(self):
        # get the basic path for a test document
        with open('data/shakespeare/shakespeare-hamlet.txt', 'rb') as f:
            text = f.read().decode(encoding='utf-8')

        vocab = sorted(set(text))

        # run the model
        try:
            model = build_model(
                vocab_size=len(vocab),
                embedding_dim=256,
                rnn_units=1024,
                batch_size=64)
        except:
            self.assertFalse(True)

        # check to make sure the model is valid
        self.assertTrue(True)

    # TODO Need to replace the None values with the correct model
    def test_train_model(self):

        # variables for the model
        model = None
        dataset = None
        epochs = 1
        checkpoint_callback = None

        # run the train_model method
        try:
            train_model(model, dataset, epochs, checkpoint_callback)
        except:
            self.assertFalse(True)

        self.assertTrue(True)

    # TODO Need to replace the None values with the correct model
    def test_generate_text(self):
        id_to_char = []
        char_to_id = {}
        seed = 'Start Text '

        with open('./shakespeare_map.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                id_to_char.append(row[1])
            char_to_id = {k: v for v, k in enumerate(id_to_char)}

        model = build_model(vocab_size=len(char_to_id), embedding_dim=256, rnn_units=1024, batch_size=64)
        model = load_model('./shakespeare_checkpoint')

        self.assertRegex(generate_text(model, seed, char_to_id, id_to_char, num_to_generate=50), "Start Text .*")


if __name__ == '__main__':
    unittest.main()
