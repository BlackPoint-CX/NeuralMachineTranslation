import tensorflow as tf
from collections import namedtuple

from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file

from iterator_utils import get_iterator
from vocab_utils import create_vocab_tables, UNK


class TrainModel(namedtuple('TrainModel', ('graph', 'model', 'iterator', 'skip_count_placeholder'))):
    pass


class EvalModel(
    namedtuple('EvalModel', ('graph', 'model', 'src_file_placeholder', 'tgt_file_placeholder', 'iterator'))):
    pass


def create_train_model_tuple(model_creator, hparams, scope=None, num_workers=1, jobid=0, extra_args=None):
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or 'train'):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab=False)

        src_dataset = tf.data.TextLineDataset(tf.gfile.Glob(hparams.src_file))
        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(hparams.tgt_file))

        skip_count_placeholder = tf.placeholder(name='skip_count_placeholder', shape=(), dtype=tf.int64)

        iterator = get_iterator(src_dataset=src_dataset,
                                tgt_dataset=tgt_dataset,
                                src_vocab_table=src_vocab_table,
                                tgt_vocab_table=tgt_vocab_table,
                                batch_size=hparams.batch_size,
                                sos=hparams.sos,
                                eos=hparams.eos,
                                src_max_len=hparams.src_max_len,
                                tgt_max_len=hparams.tgt_max_len,
                                output_buffer_size=None
                                )
        model = model_creator(hparams, mode=ModeKeys.TRAIN, iterator=iterator, source_vocab_table=src_vocab_table,
                              target_vocab_table=tgt_vocab_table,
                              reverse_target_vocab_table=None, scope=None, extra_args=None)

        return TrainModel(model=model,
                          graph=graph,
                          iterator=iterator,
                          skip_count_placeholder=skip_count_placeholder)


def create_eval_model_tuple(model_initializer, hparams, scope):
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or 'eval'):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, False)
        reverse_tgt_vocab_table = index_to_string_table_from_file(tgt_vocab_file, default_value=UNK)

        src_file = tf.placeholder(name='src_file_placeholder', type=tf.string, shape=())
        tgt_file = tf.placeholder(name='tgt_file_placeholder', type=tf.string, shape=())

        src_dataset = tf.data.TextLineDataset(tf.gfile.Glob(src_file))
        tgt_dataset = tf.data.TextLineDataset(tf.gfile.Glob(tgt_file))

        iterator = get_iterator(src_dataset=src_dataset,
                                tgt_dataset=tgt_dataset,
                                src_vocab_table=src_vocab_table,
                                tgt_vocab_table=tgt_vocab_table,
                                batch_size=hparams.batch_size,
                                sos=hparams.sos,
                                eos=hparams.eos,
                                src_max_len=hparams.src_max_len,
                                tgt_max_len=hparams.tgt_max_len,
                                output_buffer_size=None
                                )
        model = model_initializer(hparams,
                                  mode=ModeKeys.EVAL,
                                  iterator=iterator,
                                  source_vocab_table=src_vocab_table,
                                  target_vocab_table=tgt_vocab_table,
                                  reverse_target_vocab_table=None,
                                  scope=None,
                                  extra_args=None)

    return EvalModel(graph=graph,
                     model=model,
                     src_file_placeholder=src_file,
                     tgt_file_placeholder=tgt_file,
                     iterator=iterator)


def get_config_proto(log_device_placement, allow_soft_placement):
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)

    return config_proto


def load_model(model, ckpt_path, session, name):
    try:
        model.saver.restore(session, ckpt_path)
    except tf.errors.NotFoundError as e:
        print(e)

    session.run(tf.tables_initializer())
    return model


def create_or_load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

    global_step = model.global_step.eval(session=session)
    return model, global_step
