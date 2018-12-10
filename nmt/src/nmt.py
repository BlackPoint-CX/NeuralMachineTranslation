import abc
import codecs

import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, NASCell, DropoutWrapper, GRUCell, BasicLSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode, GreedyEmbeddingHelper
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
import numpy as np

from cell_utils import build_cell_list, build_rnn_cell, build_bidirectional_rnn, build_encoder_cell
from iterator_utils import BatchedInput
from other_utils import get_max_time, gradient_clip
from vocab_utils import load_embed_txt, load_vocab, _create_pretrained_emb_from_txt, _create_or_load_embed, \
    create_emb_for_encoder_and_decoder


class Model():
    def __init__(self, hparams, mode, iterator, source_vocab_table, target_vocab_table, reverse_target_vocab_table=None,
                 scope=None, extra_args=None):
        self._init_params_initializer(hparams, mode, iterator, source_vocab_table, target_vocab_table, scope,
                                      extra_args)

        res = self.build_graph(hparams, scope=scope)

        self._set_train_or_infer(res, reverse_target_vocab_table, hparams)

        self.saver = tf.train.Saver(tf.global_variables, max_to_keep=hparams.num_keep_ckpts)

    def _init_params_initializer(self, hparams, mode, iterator, source_vocab_table, target_vocab_table, scope,
                                 extra_args):
        assert isinstance(iterator, BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size

        self.time_major = hparams.time_major
        self.extra_args = extra_args

        self.num_units = hparams.num_units

        self.batch_size = tf.size(self.iterator.source_sequence_length)

        self.global_step = tf.Variable(0, trainable=False)

        self._init_embeddings(hparams, scope)

        self.dtype = hparams.dtype

    def _init_embeddings(self, hparams, scope):
        """Init embeddings."""
        self.embedding_encoder, self.embedding_decoder = create_emb_for_encoder_and_decoder(
            src_vocab_size=hparams.src_vocab_size,
            tgt_vocab_size=hparams.tgt_vocab_size,
            src_embed_size=hparams.num_units,
            tgt_embed_size=hparams.num_units,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            scope=scope)

    def build_graph(self, hparams, scope=None):

        if not self.extract_encoder_layers:
            with tf.variable_scope(scope) as scope:
                self.output_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False, name='output_projection')

        with tf.variable_scope(scope or 'dynamic_seq2seq') as scope:
            # Encode
            self.encoder_output, encoder_state = self._build_encoder(hparams)

            # Decode
            logits, decoder_cell_outputs, sample_id, final_context_state = (
                self._build_decoder(self.encoder_output, encoder_state, hparams))

            # Loss
            if self.mode != ModeKeys.PREDICT:
                loss = self._compute_loss(logits, decoder_cell_outputs)
            else:
                loss = tf.constant(0.0)
            return logits, loss, final_context_state, sample_id

    def _build_encoder(self, hparams):
        return self._build_encoder_from_sequence(hparams, self.iterator.source, self.iterator.source_sequence_length)

    def _build_encoder_from_sequence(self, hparams, sequence, sequence_length):

        num_layers = hparams.num_encoder_layers

        with tf.variable_scope('encoder') as scope:
            self.encoder_emb_inp = embedding_lookup(self.embedding_encoder, sequence)

        if hparams.encoder_type == 'uni':
            cell = build_encoder_cell(hparams)
            encoder_outputs, encoder_state = dynamic_rnn(cell=cell, inputs=self.encoder_emb_inp,
                                                         sequence_length=sequence_length)
        elif hparams.encoder_type == 'bi':
            num_bi_layers = int(num_layers / 2)
            # num_bi_residual_layers = int(num_residual_layers / 2)

            encoder_outputs, bi_encoder_state = build_bidirectional_rnn(
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

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(hparams.sos), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(hparams.eos), tf.int32)

        iterator = self.iterator

        maximum_iterations = self._get_infer_maximum_iterations(hparams, iterator.source_sequence_length)

        with tf.variable_scope('decoder') as scope:
            cell, decoder_initial_state = self._build_decoder_cell(hparams, encoder_outputs, encoder_state,
                                                                   iterator.source_sequence_length)

            logits = tf.no_op()
            decoder_cell_outputs = None

            # Train and Eval
            if self.mode != ModeKeys.PREDICT:
                target_input = iterator.target_input

                if self.time_major:
                    target_input = tf.transpose(target_input)

                decoder_emb_inp = embedding_lookup(self.tgt_vocab_table, target_input)

                helper = TrainingHelper(decoder_emb_inp, iterator.target_sequence_length, time_major=self.time_major)

                my_decoder = BasicDecoder(cell, helper, decoder_initial_state)

                outputs, final_context_state, _ = dynamic_decode(my_decoder, output_time_major=self.time_major,
                                                                 scope=scope)

                sample_id = outputs.sample_id

                logits = self.output_layer(outputs.rnn_output)

            # Inference
            else:
                infer_mode = hparams.infer_mode
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if infer_mode == 'beam_search':
                    my_decoder = None
                elif infer_mode == 'sample':
                    pass
                elif infer_mode == 'greedy':
                    helper = GreedyEmbeddingHelper(self.embedding_decoder, start_tokens, end_token)
                else:
                    raise ValueError('Unknown infer_mode %s' % infer_mode)

                if infer_mode != 'beam_search':
                    my_decoder = BasicDecoder(cell, helper, decoder_initial_state, output_layer=self.output_layer)

                outputs, final_context_state, _ = dynamic_decode(my_decoder, maximum_iterations=maximum_iterations,
                                                                 output_time_major=self.time_major, scope=scope)

        return logits, decoder_cell_outputs, sample_id, final_context_state

    def _compute_loss(self, logits, decoder_cell_outputs):
        target_output = self.iterator.target_output

        if self.time_major:
            target_output = tf.transpose(target_output)

        max_time = get_max_time(target_output)

        corssent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_output)

        target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=self.dtype)

        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(corssent * target_weights) / tf.to_float(self.batch_size)

        return loss

    def _set_train_or_infer(self, res, reverse_target_vocab_table, hparams):
        # logits, loss, final_context_state, sample_id
        if self.mode == ModeKeys.TRAIN:
            self.train_loss = res[1]
        elif self.mode == ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif self.mode == ModeKeys.PREDICT:
            self.infer_logits, self.infer_loss, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(tf.to_int64(self.sample_id))

        if self.mode == ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)

            self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps,
                                                            decay_rate=hparams.decay_rate)

            if hparams.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif hparams.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError('Unknown optimizer %s' % hparams.optimizer)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, params)

            self.clipped_grads, self.grad_norm_summary, self.grad_norm = \
                gradient_clip(gradients, max_gradient_norm=hparams.max_gradient_norm)

            self.update = opt.apply_gradients(zip(self.clipped_grads, params), global_step=self.global_step)
            self.train_summary = self._get_train_summary()

    def _get_train_summary(self):
        train_summary = tf.summary.merge([tf.summary.scalar('lr'), self.learning_rate,
                                          tf.summary.scalar('train_loss', self.train_loss)] +
                                         self.grad_norm_summary
                                         )
        return train_summary
