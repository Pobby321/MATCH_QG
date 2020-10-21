import sys

sys.path.insert(0, '../')

import config
from data_utils import (make_conll_format, make_embedding, make_vocab,
                        make_vocab_from_squad, process_file)




def make_sent_dataset():
    train_src_file = "./para-train.txt"
    train_trg_file = "./tgt-train.txt"

    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    word2idx_file = "./word2idx.pkl"
    # make vocab file
    word2idx = make_vocab(train_src_file, train_trg_file, word2idx_file, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)


def make_para_dataset():
    embedding_file = "word2vec_baike/word2vec_baike"
    embedding = "word2vec_baike/embedding.pkl"
    src_word2idx_file = "word2vec_baike/word2idx.pkl"

    train_squad = "data/round1_train_0907.json"
    test_squad = "data/round1_test_0907.json"

    train_src_file = "data/para-train.txt"
    train_trg_file = "data/tgt-train.txt"
    dev_src_file = "data/para-dev.txt"
    dev_trg_file = "data/tgt-dev.txt"

    test_src_file = "data/para-test.txt"
    test_trg_file = "data/tgt-test.txt"

    # pre-process training data
    train_examples, counter = process_file(train_squad)

    word2idx = make_vocab_from_squad(src_word2idx_file, counter, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    # split dev into dev and test
    test_examples, _ = process_file(test_squad)
    # random.shuffle(dev_test_examples)
    num_dev = int(len(train_examples)*0.8)
    train_new_examples = train_examples[:num_dev]
    dev_examples = train_examples[num_dev:]
    make_conll_format(train_new_examples, train_src_file, train_trg_file)
    make_conll_format(dev_examples, dev_src_file, dev_trg_file)
    make_conll_format(test_examples, test_src_file, test_trg_file)


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()
