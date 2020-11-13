import json


class Corpus(object):

    def __init__(self, corpus_file, corpus_dir):
        with open(corpus_file) as json_file:
            data = json.load(json_file)
        self.data = data['directory']
        self.dir = corpus_dir

    def full_file_list(self):
        return ['{}{}'.format(self.dir, elem['filename']) for elem in self.data]

    def full_combined_string(self):
        combined = ''
        for elem in self.full_file_list():
            with open(elem) as f:
                combined += '{}\n'.format(f.read())
        return combined

    def author_file_list(self, author):
        return ['{}{}'.format(self.dir, elem['filename']) for elem in self.data if elem['author'] == author]

    def author_combined_string(self, author):
        combined = ''
        for elem in self.author_file_list(author):
            with open(elem) as f:
                combined += '{}\n'.format(f.read())
        return combined

    def tag_file_list(self, tag):
        return ['{}{}'.format(self.dir, elem['filename']) for elem in self.data if tag in elem['tags']]

    def tag_combined_string(self, tag):
        combined = ''
        for elem in self.tag_file_list(tag):
            with open(elem) as f:
                combined += '{}\n'.format(f.read())
        return combined


if __name__ == '__main__':
    test = Corpus('./data/corpus_directory.json', './data/corpus/')
