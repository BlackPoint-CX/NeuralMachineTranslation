import codecs
import numpy as np
import tensorflow as tf
from tensorflow.contrib.lookup import lookup_ops

UNK_ID = 0
SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


def load_embed_txt(embed_file):
    """
    载入预训练的embedding文件. Glove格式或Word2Vec格式.
    :param embed_file:
    :return: emb_dict: Map(word,list(float), emb_size:int
    """
    emb_dict = dict()
    emb_size = None

    is_first_line = True

    with codecs.getreader('utf-8')(tf.gfile.GFile(embed_file), 'rb') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if is_first_line:
                is_first_line = False
                if len(tokens) == 2:
                    emb_size = int(tokens[1])
                continue
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec

            if emb_size:
                if emb_size != len(vec):
                    del emb_dict[word]
            else:
                emb_size = len(vec)

    return emb_dict, emb_size


def load_vocab(vocab_file):
    """
    从文件中载入词典.
    :param vocab_file:
    :return:
    """
    vocab = []
    vocab_size = 0
    with codecs.getreader(encoding='utf-8')(tf.gfile.GFile(vocab_file)) as f:
        for line in f:
            vocab.append(line.strip())
            vocab_size += 1

    return vocab, vocab_size


def _create_pretrained_emb_from_txt(vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32):
    vocab, vocab_size = load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]
    emb_dict, emb_size = load_embed_txt(embed_file)

    # 对于预训练中没有存在的词, 通过初始化进行赋值.
    for token in trainable_tokens:
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size

    emb_mat = np.array([emb_dict[token] for token in vocab],
                       dtype=dtype.as_numpy_dtype)  # 这一步可能会存在问题, 因为可能有的词在embedding文件中没有.
    emb_mat = tf.constant(emb_mat)

    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])  # 把除了trainable_tokens的部分切割出来

    with tf.variable_scope('pretrain_embeddings', dtype=dtype) as scope:
        emb_mat_var = tf.get_variable(name='emb_mat_var', shape=[num_trainable_tokens, emb_size], dtype=dtype)

    return tf.concat([emb_mat_var, emb_mat_const], 0)


def _create_or_load_embed(embed_name, vocab_file, embed_file, vocab_size, embed_size, dtype):
    if vocab_file and embed_file:
        embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
    else:
        embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)

    return embedding


def create_emb_for_encoder_and_decoder(src_vocab_size, tgt_vocab_size, src_embed_size,
                                       tgt_embed_size, src_vocab_file, tgt_vocab_file,
                                       src_embed_file, tgt_embed_file, scope, dtype=tf.float32):
    with tf.variable_scope(scope or 'embeddings') as scope:
        embedding_encoder = _create_or_load_embed('embedding_encoder', src_vocab_file, src_embed_file,
                                                  src_vocab_size, src_embed_size, dtype)
        embedding_decoder = _create_or_load_embed('embedding_decoder', tgt_vocab_file, tgt_embed_file,
                                                  tgt_vocab_size, tgt_embed_size, dtype)

    return embedding_encoder, embedding_decoder


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table
