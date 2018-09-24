import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper, BahdanauAttention, AttentionWrapper, \
    BasicDecoder, dynamic_decode
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2


def seq2seq(mode, features, labells, params):
    vocab_size = params['vocab_size']  # 传入的词典大小
    embed_dim = params['embed_dim']  # 词向量大小
    num_units = params['num_units']
    input_max_length = params['input_max_length']
    output_max_length = params['output_max_length']

    inp = features['input']
    output = features['output']
    batch_size = tf.shape(inp)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], axis=1)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, 1)), axis=1)
    output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), axis=1)
    input_embed = layers.embed_sequence(ids=inp, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    # input_embed = tf.nn.embedding_lookup(params=embed_dim)
    output_embed = layers.embed_sequence(ids=train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed',
                                         reuse=True)
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    cell = GRUCell(num_units=num_units)
    encoder_outputs, encoder_state = dynamic_rnn(cell=cell, inputs=input_embed, sequence_length=input_lengths,
                                                 dtype=tf.float32)
    train_helper = TrainingHelper(inputs=output_embed, sequence_length=output_lengths)

    pred_helper = GreedyEmbeddingHelper(embedding=embeddings, start_tokens=tf.to_int32(start_tokens), end_token=1)

    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, resue=reuse):
            attention_mechnism = BahdanauAttention(num_units=num_units, memory=encoder_outputs,
                                                   memory_sequence_length=input_lengths)
            cell = GRUCell(num_units=num_units)
            attn_cell = AttentionWrapper(cell=cell, attention_mechanism=attention_mechnism,
                                         attention_layer_size=num_units / 2)
            out_cell = OutputProjec
            tionWrapper(cell=attn_cell, output_size=vocab_size, reuse=reuse)
            decoder = BasicDecoder(cell=out_cell, helper=helper,
                                   initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))
            outputs = dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True,
                                     maximum_iterations=output_max_length)

            return outputs[0]

    def decode(helper,scope,reuse=None):
        with tf.variable_scope(scope,reuse=reuse):
            attention_mechanism = BahdanauAttention(num_units=num_units,memory=encoder_outputs,memory_sequence_length=input_lengths)
            cell= GRUCell(num_units=num_units)
            atten_cell = AttentionWrapper(cell=cell,attention_mechanism=attention_mechanism,attention_layer_size=num_units/2)
