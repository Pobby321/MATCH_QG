import config
import numpy as np
# from torch_scatter import scatter_max
from data_utils import UNK_ID
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
INF = 1e12

"""
class Encoder():
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):

        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_layer = nn.Linear(
            4 * hidden_size, 2 * hidden_size, bias=False)
        self.gate = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        mask = mask.unsqueeze(1)
        energies = energies.masked_fill(mask == 0, value=-1e12)

        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def forward(self, src_seq, src_len, tag_seq):
        total_length = src_seq.get_shape().as_list()[1]
        embedded = self.embedding(src_seq)
        tag_embedded = self.tag_embedding(tag_seq)
        embedded = torch.cat((embedded, tag_embedded), dim=2)
        packed = pack_padded_sequence(embedded,
                                      src_len,
                                      batch_first=True,
                                    )
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs,
                                         batch_first=True,
                                         total_length=total_length)  # [b, t, d]
        h, c = states

        # self attention
        mask = torch.sign(src_seq)
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size,
                 embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(
            embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask == 0, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        device = trg_seq.device
        batch_size, max_len = trg_seq.size()

        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        # init decoder hidden states and context vector
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size))
        prev_context = prev_context.to(device)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(
                torch.cat([embedded, prev_context], 2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov))
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
        output, states = self.lstm(lstm_inputs, prev_states)

        context, energy = self.attention(output,
                                         encoder_features,
                                         encoder_mask)
        concat_input = torch.cat((output, context), 2).squeeze(1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -INF

        return logit, states, context
"""

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

        # self.encoder = Encoder(embedding,
        #                        config.vocab_size,
        #                        config.embedding_size,
        #                        config.hidden_size,
        #                        config.num_layers,
        #                        config.dropout)
        # self.decoder = Decoder(embedding, config.vocab_size,
        #                        config.embedding_size,
        #                        2 * config.hidden_size,
        #                        config.num_layers,
        #                        config.dropout)
        enc_mask = tf.sign(self.src_seq)
        src_len = tf.reduce_sum(enc_mask, 1)
        enc_outputs, enc_states = self.encoder(self.src_seq, src_len, self.tag_seq)
        sos_trg = self.trg_seq[:, :-1]

        self.logits = self.decode(sos_trg, self.ext_src_seq,
                              enc_states, enc_outputs, enc_mask)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sos_trg,logits=self.logits)
        self.opt = tf.train.AdamOptimizer(config['lr'])
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

        # packed = pack_padded_sequence(embedded,
        #                               src_len,
        #                               batch_first=True,
        #                               )
        # outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        # outputs, _ = pad_packed_sequence(outputs,
        #                                  batch_first=True,
        #                                  total_length=total_length)  # [b, t, d]
        outputs = tf.concat(outputs_,-1)
        h, c = states

        # self attention
        mask = tf.sign(src_seq)
        memories = tf.layers.dense(outputs,2*self.hidden_size)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.get_shape().as_list()
        h = tf.reshape(h,shape=(2, 2, b, d))
        # h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = tf.concat((h[:, 0, :, :], h[:, 1, :, :]), axis=-1)

        c = tf.reshape(c,shape=(2, 2, b, d))
        c = tf.concat((c[:, 0, :, :], c[:, 1, :, :]), axis=-1)
        concat_states = (h, c)

        return outputs, concat_states

    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = tf.matmul(query,tf.transpose(memories,shape=(1, 2))) # [b, 1, t]
        energy = tf.squeeze(energy,1)
        energy =energy - (1 - mask)*INF
        attn_dist = tf.tile(tf.nn.softmax(energy, dim=1),1)  # [b, 1, t]
        context_vector = tf.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return tf.layers.dense(encoder_outputs,self.hidden_size)

    def decode(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]

        batch_size, max_len = tf.shape(trg_seq)

        hidden_size = encoder_outputs.get_shape().as_list()[-1]
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        # init decoder hidden states and context vector
        prev_states = init_states
        prev_context = tf.zeros((batch_size, 1, hidden_size))
        for i in range(max_len):
            y_i = tf.tile(trg_seq[:, i],1) # [b, 1]
            embedded = tf.nn.embedding_lookup(self.embedding,y_i)
            lstm_inputs = tf.layers.dense(
                tf.concat([embedded, prev_context], 2),self.embedding_size)
            f_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            b_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            outputs_, states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, lstm_inputs,init_states=prev_states)
            output = tf.concat(outputs_,2)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = tf.squeeze(tf.concat((output, context), axis=2),1)
            logit_input = tf.tanh(tf.layers.dense(concat_input,self.hidden_size))
            logit = tf.layers.dense(logit_input,self.vocab_size)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(tf.reduce_max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = tf.zeros((batch_size, num_oov))
                extended_logit = tf.concat([logit, zeros], dim=1)
                out = tf.zeros_like(extended_logit) - INF
                out = tf.compat.v1.scatter_max(out,energy, ext_src_seq)
                # out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = tf.where(tf.equal(out,-INF),tf.zeros_like(out),out)
                logit = extended_logit + out
                tf.where(tf.equal(logit, 0), tf.ones_like(logit)*(-INF), logit)
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = tf.stack(logits, dim=1)  # [b, t, |V|]


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
