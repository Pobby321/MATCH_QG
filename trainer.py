import os
import pickle
import time

import numpy as np
from util import get_logger, create_model,save_model , make_path
import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq
import tensorflow as tf


class Trainer(object):
    def __init__(self, args):
        # load dictionary and embedding file
        with open(config.embedding, "rb") as f:
            embedding = np.array(pickle.load(f))
        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       use_tag=True,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     use_tag=True,
                                     batch_size=128,
                                     debug=config.debug)

        make_path(config)
        log_path = os.path.join("log", config.log_file)
        self.logger = get_logger(log_path)


    def train(self):
        batch_num = len(self.train_loader)
        best_loss = 1e10
        with tf.Session() as sess:
            model = create_model(sess, Seq2seq, config.ckpt_path, config, self.logger)

            self.logger.info("start training")
            loss = []
            for epoch in range(1, config.num_epochs + 1):
                start = time.time()

                for batch_idx, batch in enumerate(self.train_loader.iter(), start=1):
                    step, batch_loss = model.run_step(sess, True, batch)

                    loss.append(batch_loss)

                    msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                        .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                                eta(start, batch_idx, batch_num), batch_loss)
                    print(msg, end="\r")

                val_loss = self.evaluate(msg)
                if val_loss <= best_loss:
                    best_loss = val_loss
                    save_model(val_loss, epoch)

                print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                      .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))



    def evaluate(self, msg):
        self.model.eval()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(
                    msg, i, num_val_batches)
                print(msg2, end="\r")
        
        val_loss = np.mean(val_losses)

        return val_loss
