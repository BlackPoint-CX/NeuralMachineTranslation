import abc

import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, NASCell, DropoutWrapper, GRUCell, BasicLSTMCell, MultiRNNCell
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn


class BaseModel():
    def __init__(self):
        pass

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        pass

    def _build_single_cell(self, mode, unit_type, num_units, forget_bias, dropout):
        dropout = dropout if mode == ModeKeys.TRAIN else 0.0

        unit_type = unit_type.lower()
        assert unit_type in ['lstm', 'gru', 'layer_norm_lstm', 'nas']
        if unit_type == 'lstm':
            cell = BasicLSTMCell(num_units, forget_bias=forget_bias)
        elif unit_type == 'gru':
            cell = GRUCell(num_units)
        elif unit_type == 'layer_norm_lstm':
            cell = LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias)
        elif unit_type == 'nas':
            cell = NASCell(num_units)
        else:
            raise ValueError('Unknow unit_type %s !' % unit_type)

        if dropout > 0.0:
            cell = DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))

        return cell

    def _build_cell_list(self, num_layers, mode, unit_type, num_units, forget_bias, dropout, single_cell_fn=None):
        if not single_cell_fn:
            single_cell_fn = self._build_single_cell

        cell_list = []

        for i in range(num_layers):
            single_cell = single_cell_fn(mode=mode,
                                         unit_type=unit_type,
                                         num_units=num_units,
                                         forget_bias=forget_bias,
                                         dropout=dropout)
            cell_list.append(single_cell)

        return cell_list

    def _build_rnn_cell(self, num_layers, mode, unit_type, num_units, forget_bias, dropout, single_cell_fn=None):
        cell_list = self._build_cell_list(num_layers, mode, unit_type, num_units, forget_bias, dropout,
                                          single_cell_fn=None)
        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return MultiRNNCell(cells=cell_list)

    def _init_embeddings(self, hparams, scope):
        """Init embeddings."""
        self.embedding_encoder, self.embedding_decoder = (
            self.create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=self.num_units,
                tgt_embed_size=self.num_units,
                num_enc_partitions=hparams.num_enc_emb_partitions,
                num_dec_partitions=hparams.num_dec_emb_partitions,
                src_vocab_file=hparams.src_vocab_file,
                tgt_vocab_file=hparams.tgt_vocab_file,
                src_embed_file=hparams.src_embed_file,
                tgt_embed_file=hparams.tgt_embed_file,
                use_char_encode=hparams.use_char_encode,
                scope=scope, ))

    def build_graph(self, hparams, scope=None):
        with tf.variable_scope(scope or 'dynamic_seq2seq') as scope:
            # Encode
            self.encoder_output, encoder_state = self._build_encoder(hparams)

            # Decode
            logits, decoder_cell_outputs, sample_id, final_context_state = (
                self._build_decoder(self.encoder_outputs, encoder_state, hparams))

            # Loss
            if self.mode != ModeKeys.PREDICT:
                loss = self._compute_loss(logits, decoder_cell_outputs)
            else:
                loss = tf.constant(0.0)
            return logits, loss, final_context_state, sample_id


class Model(BaseModel):

    def _build_encoder(self, hparams):
        return self._build_encoder_from_sequence(hparams, self.iterator.source, self.iterator.source_sequence_length)

    def _build_decoder(self):
        pass

    def _build_encoder_cell(self, hparmas):
        return self._build_rnn_cell(num_layers=hparmas.num_layers,
                                    mode=hparmas.mode,
                                    unit_type=hparmas.unit_type,
                                    num_units=hparmas.num_units,
                                    forget_bias=hparmas.forget_bias,
                                    dropout=hparmas.dropout,
                                    single_cell_fn=hparmas.single_cell_fn
                                    )

    def _build_decoder_cell(self):
        pass

    def _build_bidirectional_rnn(self, inputs, sequence_length, dtype, hparams, num_bi_layers=None,
                                 num_bi_residual_layers=None):
        fw_cell = self._build_encoder_cell(hparams)
        bw_cell = self._build_encoder_cell(hparams)
        bi_output, bi_state = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
                                                        sequence_length=sequence_length)
        encoder_outputs = tf.concat(bi_output, -1)
        encoder_state = bi_state
        return encoder_outputs, encoder_state

    def _build_encoder_from_sequence(self, hparams, sequence, sequence_length):

        num_layers = hparams.num_encoder_layers
        # num_residual_layers = self.num_encoder_residual_layers

        with tf.variable_scope('encoder') as scope:
            self.encoder_emb_inp = embedding_lookup(self.embedding_encoder, sequence)

        if hparams.encoder_type == 'uni':
            cell = self._build_encoder_cell(hparams)
            encoder_outputs, encoder_state = dynamic_rnn(cell=cell, inputs=self.encoder_emb_inp,
                                                         sequence_length=sequence_length)

        elif hparams.encoder_type == 'bi':
            num_bi_layers = int(num_layers / 2)
            # num_bi_residual_layers = int(num_residual_layers / 2)

            encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                inputs=self.encoder_emb_inp,
                sequence_length=sequence_length,
                hparams=hparams)

            if num_bi_layers == 1:
                encoder_state = bi_encoder_state
            else:
                # alternatively concat forward and backward states
                encoder_state = []
                for layer_id in range(num_bi_layers):
                    encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                    encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                encoder_state = tuple(encoder_state)

        else:
            raise ValueError('Unknow encoder_type %s !' % hparams.encoder_type)

        return encoder_outputs, encoder_state
