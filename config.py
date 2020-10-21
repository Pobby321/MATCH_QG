# train file
train_src_file = "data/para-train.txt"
train_trg_file = "data/tgt-train.txt"
# dev file
dev_src_file = "data/para-dev.txt"
dev_trg_file = "data/tgt-dev.txt"
# test file
test_src_file = "data/para-test.txt"
test_trg_file = "data/tgt-test.txt"
# embedding and dictionary file
embedding = "word2vec_baike/embedding.pkl"
word2idx_file = "./word2vec_baike/word2idx.pkl"

ckpt_path = "ckpt"
log_file = "train.txt"
result_path = 'result'

device = "cuda:1"
use_gpu = True
debug = False
vocab_size = 45000
freeze_embedding = True

num_epochs = 20
max_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
batch_size = 64
dropout_keep = 0.3
max_grad_norm = 5.0

use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer_maxout_ans"
