import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BasicDecoder, dynamic_decode, \
    BeamSearchDecoder
from tensorflow.contrib.training import HParams
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import DeviceWrapper, LSTMCell, MultiRNNCell

from base_model import BaseModel


class NeuralMachineTranslation(BaseModel):
    def __init__(self, config):
        super(NeuralMachineTranslation, self).__init__(config)
        pass

    def add_placeholder_op(self):
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='encoder_inputs')
        self.source_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, None], name='source_sequence_length')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='decoder_outputs')
        self.result_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, None], name='result_sequence_length')

    def add_encoder_embedding_op(self):
        word_embedding_shape = [self.config.source_word_vocab_size, self.config.source_word_embedding_dim]
        # Embedding
        if self.config.use_pretrain_embedding is None:
            self.source_embedding_encoder = tf.get_variable(name='embedding_encoder', shape=word_embedding_shape,
                                                            dtype=tf.float32,
                                                            initializer=tf.random_normal_initializer())
        else:
            self.source_embedding_encoder = tf.Variable(initial_value=self.config.source_pretrain_embedding,
                                                        name='source_pretrain_embedding',
                                                        dtype=tf.float32, expected_shape=word_embedding_shape,
                                                        trainable=self.config.trainable)

        self.encoder_emb_inp = embedding_lookup(params=self.source_embedding_encoder, ids=self.encoder_inputs)

    def add_encoder_op(self):
        if self.config.cell_type == 'lstm':
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units)
        elif self.config.cell_type == 'gru':
            encoder_cell = tf.nn.rnn_cell.GRUCell(self.config.num_units)
        else:
            raise NotImplementedError('Cell Type should be in [lstm, gru]')
        self.encoder_outputs, self.encoder_state = dynamic_rnn(cell=encoder_cell,
                                                               inputs=self.encoder_emb_inp,
                                                               sequence_lengths=self.source_sequence_length,
                                                               dtype=tf.float32,
                                                               time_major=True)

    def add_decoder_op(self):
        # Helper
        helper = TrainingHelper(inputs=self.decoder_emb_inp, sequence_length=self.decoder_lengths, time_major=True)
        # Decoder
        if self.config.cell_type == 'lstm':
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_units)
        elif self.config.cell_type == 'gru':
            decoder_cell = tf.nn.rnn_cell.GRUCell(self.config.num_units)
        else:
            raise NotImplementedError('Cell Type should be in [lstm, gru]')
        self.decoder = BasicDecoder(cell=decoder_cell,
                                    helper=helper,
                                    initial_state=self.encoder_state,
                                    output_layer=self.projection_layer)

        # Dynamic Decoding
        self.decoder_outputs, self.decoder_state = dynamic_decode(decoder=self.decoder, output_time_major=True)
        self.logits = self.decoder_outputs.rnn_output

    def add_projection_op(self):
        self.projection_layer = Dense(self.config.tgt_vocab_size, use_bias=False)

    def add_loss_op(self):
        target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=self.dtype)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs, logits=self.logits)
        train_loss = (tf.reduce_sum(cross_entropy * self.target_weights) / self.config.batch_size)

    def add_inference_op(self):
        # Helper
        helper = GreedyEmbeddingHelper(embedding=self.embedding_decoder,
                                       start_tokens=tf.fill([self.config.batch_size], self.config.tgt_sos_id),
                                       end_token=self.config.tgt_eos_id)
        # Decoder
        decoder = BasicDecoder(cell=self.decoder_cell, helper=helper, initial_state=self.encoder_state,
                               output_layer=self.projection_layer)
        # Dynamic Decoding
        maximum_iterations = tf.round(tf.reduce_max(self.source_sequence_length) * 2)
        outputs, _ = dynamic_decode(decoder=decoder, output_time_major=True, maximum_iterations=maximum_iterations)
        self.translations = outputs.sample_id

        BeamSearchDecoder()

        cells = []
        for i in range(num_layers):
            cell = DeviceWrapper(cell=LSTMCell(num_units), device='/gpu:%d' % (num_layers % num_gpus))
            cells.append()
        cell = MultiRNNCell(cells)
