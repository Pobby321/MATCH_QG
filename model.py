import config
import numpy as np
from data_utils import UNK_ID
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
INF = 1e12

class Seq2seq():
    def __init__(self, config):
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.max_len = config.max_len
        self.global_step = tf.Variable(0, trainable=False)
        self.src_seq = tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.tag_seq = tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.ext_src_seq = tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.trg_seq = tf.placeholder(dtype=tf.int32,shape=[None,self.max_len])
        self.dropout = tf.placeholder(dtype=tf.float32)

        self.initializer = initializers.xavier_initializer()
        self.embedding = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size), name='encoder_embedding',initializer=self.initializer)
        self.tag_embedding = tf.get_variable(dtype=tf.float32, shape=(3, 3), name='tag_embedding',initializer=self.initializer)
        lstm_input_size = self.embedding_size + 3

        enc_mask = tf.sign(self.src_seq)
        src_len = tf.reduce_sum(enc_mask, 1)
        enc_outputs, enc_states = self.encoder(self.src_seq, src_len, self.tag_seq)
        sos_trg = self.trg_seq[:, :-1]

        self.logits = self.decode(sos_trg, self.ext_src_seq,
                              enc_states, enc_outputs, enc_mask)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sos_trg,logits=self.logits)

        self.opt = tf.train.AdamOptimizer(config.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.train_op = self.opt.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # def focal_loss(y_true, y_pred):
    #     alpha, gamma = 0.25, 2
    #     y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
    #     # y_pred = tf.squeeze(y_pred,2)
    #     return - alpha * y_true * tf.log(y_pred) * (1 - y_pred) ** gamma - (1 - alpha) * (1 - y_true) * tf.log(
    #         1 - y_pred) * y_pred ** gamma

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        energies = tf.matmul(queries, tf.transpose(memories,(0, 2, 1)))  # [b, t, t]
        mask = tf.expand_dims(energies,1)
        energies = tf.where(tf.equal(energies, 0), tf.ones_like(energies)*(-INF), energies)
        # energies = energies.masked_fill(mask == 0, value=-1e12)

        scores = tf.nn.softmax(energies, axis=2)
        context = tf.matmul(scores, queries)
        inputs = tf.concat([queries, context], axis=2)
        f_t = tf.tanh(tf.layers.dense(inputs,2*self.hidden_size))
        g_t = tf.sigmoid(tf.layers.dense(inputs,2*self.hidden_size))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def encoder(self, src_seq, src_len, tag_seq):
        total_length = src_seq.get_shape().as_list()[1]
        embedded = tf.nn.embedding_lookup(self.embedding,src_seq)
        tag_embedded =  tf.nn.embedding_lookup(self.tag_embedding,tag_seq)
        embedded = tf.concat((embedded, tag_embedded), axis=2)
        f_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        b_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        outputs_, states = tf.nn.bidirectional_dynamic_rnn(f_cell,b_cell,embedded,dtype=tf.float32)
        # _inputs = embedded
        # for _ in range(self.num_layers):
        #     # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
        #     # 恶心的很.如果不在这加的话,会报错的.
        #     with tf.variable_scope(None, default_name="bidirectional-rnn"):
        #         f_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        #         b_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        #         (output, state) = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, _inputs,dtype=tf.float32)
        #         _inputs = tf.concat(output, 2)
        # cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        # cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        # initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        # initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        # outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
        #                                                     initial_states_fw=initial_states_fw,
        #                                                     initial_states_bw=initial_states_bw, dtype=tf.float32)
        outputs = tf.concat(outputs_,-1)
        (f_c, f_h),(b_c,b_h) = states

        # self attention
        mask = tf.sign(src_seq)
        memories = tf.layers.dense(outputs,2*self.hidden_size)
        outputs = self.gated_self_attn(outputs, memories, mask)

        # _, b, d = h.get_shape().as_list()
        # h = tf.reshape(h,shape=(2, 2, b, d))
        # # h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        # h = tf.concat((h[:, 0, :, :], h[:, 1, :, :]), axis=-1)
        #
        # c = tf.reshape(c,shape=(2, 2, b, d))
        # c = tf.concat((c[:, 0, :, :], c[:, 1, :, :]), axis=-1)
        concat_states = (f_h, b_h)

        return outputs, states

    def attention(self,query, memories, mask):
        # query : [b, 1, d]
        energy = tf.matmul(query,tf.transpose(memories,perm=(0,2,1))) # [b, 1, t]
        energy = tf.squeeze(energy,1)
        # energy =energy - tf.multiply((1 - mask),INF)
        energy = tf.where(tf.equal(mask,1),energy,tf.zeros_like(energy)*INF)
        attn_dist = tf.expand_dims(tf.nn.softmax(energy, axis=1),1)  # [b, 1, t]
        context_vector = tf.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return tf.layers.dense(encoder_outputs,self.hidden_size*2)

    def decode(self, trg_seq, ext_src_seq, init_state, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]

        batch_size, max_len = trg_seq.get_shape().as_list()

        hidden_size = encoder_outputs.get_shape().as_list()[-1]
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        # init decoder hidden states and context vector
        prev_states = init_state
        prev_context = tf.zeros((tf.shape(trg_seq)[0], 1, hidden_size))
        for i in range(max_len):
            y_i = tf.expand_dims(trg_seq[:, i],1) # [b, 1]
            embedded = tf.nn.embedding_lookup(self.embedding,y_i)
            lstm_inputs = tf.layers.dense(
                tf.concat([embedded, prev_context], 2),self.embedding_size)
            f_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,name='cell_fw_{}'.format(i))
            b_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size,name='cell_bw_{}'.format(i))
            outputs_, states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, lstm_inputs,initial_state_fw=prev_states[0],initial_state_bw=prev_states[1])
            output = tf.concat(outputs_,2)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = tf.squeeze(tf.concat((output, context), axis=2),1)
            logit_input = tf.tanh(tf.layers.dense(concat_input,self.hidden_size))
            logit = tf.layers.dense(logit_input,self.vocab_size)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = tf.reduce_max(tf.reduce_max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = tf.zeros((batch_size, num_oov))
                extended_logit = tf.concat([logit, zeros], axis=1)
                out = tf.zeros_like(extended_logit) - INF
                out = tf.compat.v1.scatter_max(out,energy, ext_src_seq)
                # out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = tf.where(tf.equal(out,-INF),tf.zeros_like(out),out)
                logit = extended_logit + out
                tf.where(tf.equal(logit, 0), tf.ones_like(logit)*(-INF), logit)
                logit = tf.where(tf.equal(logit,0),)
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = tf.stack(logits, axis=1)  # [b, t, |V|]


        return logits


    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        src_seq, ext_src_seq, trg_seq, ext_trg_seq, tag_seq = batch
        feed_dict = {
            self.src_seq: np.asarray(src_seq),
            self.tag_seq: np.asarray(tag_seq),
            self.ext_src_seq: np.asarray(ext_src_seq),
            self.dropout: 1.0,
        }
        if is_train:
            self.trg_seq: np.asarray(trg_seq)

            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([ self.logits], feed_dict)
            return lengths, logits
