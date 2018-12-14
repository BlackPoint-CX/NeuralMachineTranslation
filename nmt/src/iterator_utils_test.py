import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.ops.lookup_ops import index_table_from_tensor

from iterator_utils import get_iterator


class IteratorUtilsTest(tf.test.TestCase):

    def testGetIterator(self):
        tf.set_random_seed(1)
        src_vocab_table = tgt_vocab_table = index_table_from_tensor(tf.constant(["a", "b", "c", "eos", "sos"]))

        src_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(["f e a g", "c c a", "d", "c a"]))
        tgt_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(["c c", "a b", "", "b c"]))

        hparams = HParams(random_seed=3, num_buckets=5, eos='eos', sos='sos')

        batch_size = 2

        src_max_len = 3
        tgt_max_len = 3

        iterator = get_iterator(src_dataset=src_dataset,
                                tgt_dataset=tgt_dataset,
                                src_vocab_table=src_vocab_table,
                                tgt_vocab_table=tgt_vocab_table,
                                batch_size=batch_size,
                                sos=hparams.sos,
                                eos=hparams.eos,
                                src_max_len=src_max_len,
                                tgt_max_len=tgt_max_len)

        table_initializer = tf.tables_initializer()
        source = iterator.source
        target_input = iterator.target_input
        target_output = iterator.target_output
        src_seq_len = iterator.source_sequence_length
        tgt_seq_len = iterator.target_sequence_length

        self.assertEqual([None, None], source.shape.as_list())  # 判断输出的source shape
        self.assertEqual([None, None], target_input.shape.as_list())
        self.assertEqual([None, None], target_output.shape.as_list())

        with self.test_session() as sess:
            sess.run(table_initializer)
            sess.run(iterator.initializer)

            (source_v, src_len_v, target_input_v, target_output_v, tgt_len_v) = (
                sess.run((source, src_seq_len, target_input, target_output,
                          tgt_seq_len)))

            print(source_v)
            self.assertAllEqual(
                [[-1, -1, 0],  # "f" == unknown, "e" == unknown, a
                 [2, 2, 0]],  # c a eos -- eos is padding
                source_v)
