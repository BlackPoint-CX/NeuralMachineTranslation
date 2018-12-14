from collections import namedtuple
import tensorflow as tf


class BatchedInput(namedtuple(typename='BatchedInput',
                              field_names=('initializer', 'source', 'target_input', 'target_output',
                                           'source_sequence_length', 'target_sequence_length'))):
    pass


def get_iterator(src_dataset, tgt_dataset,
                 src_vocab_table, tgt_vocab_table,
                 batch_size,
                 sos, eos,
                 src_max_len, tgt_max_len,
                 output_buffer_size=None,
                 num_parallel_calls=4,
                 skip_count=None):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000

    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[:src_max_len], tgt),
                                              num_parallel_calls=num_parallel_calls)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src, tgt[:tgt_max_len]),
                                              num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                                            tf.cast(src_vocab_table.lookup(src), tf.int32)),
                                          num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src,
                                                            tf.concat(([tgt_sos_id], tgt), 0),
                                                            tf.concat((tgt, [tgt_eos_id]), 0)),
                                          num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_inp, tgt_out: (src, tgt_inp, tgt_out, tf.size(src), tf.size(tgt_inp)))

    batched_dataset = src_tgt_dataset.padded_batch(batch_size=batch_size,
                                                   padded_shapes=(
                                                       tf.TensorShape([None]),
                                                       tf.TensorShape([None]),
                                                       tf.TensorShape([None]),
                                                       tf.TensorShape([]),
                                                       tf.TensorShape([])),
                                                   padding_values=(
                                                       src_eos_id,
                                                       tgt_sos_id,
                                                       tgt_eos_id,
                                                       0,
                                                       0))

    batch_iterator = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_inp_ids, tgt_out_ids, src_seq_len, tgt_seq_len) = (batch_iterator.get_next())

    return BatchedInput(initializer=batch_iterator.initializer,
                        source=src_ids,
                        target_input=tgt_inp_ids,
                        target_output=tgt_out_ids,
                        source_sequence_length=src_seq_len,
                        target_sequence_length=tgt_seq_len)
