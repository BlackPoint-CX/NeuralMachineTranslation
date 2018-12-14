import math
import os
import sys
import time
from argparse import ArgumentParser
from tensorflow.python import debug as tf_debug
import tensorflow as tf

from model_builder_utils import create_train_model_tuple, get_config_proto, create_or_load_model, \
    create_eval_model_tuple
from other_utils import safe_exp
from param_utils import add_parameter, create_hparams
from standard_model import StandardModel


def get_model_creator(hparams):
    if hparams.model_type == 'standard':
        model_creator = StandardModel

    else:
        raise ValueError('Unknown model_type %s ' % hparams.model_type)

    return model_creator


def init_stats():
    """Initialize statistics that we want to accumulate."""
    return {"step_time": 0.0, "train_loss": 0.0,
            "predict_count": 0.0,  # word count on the target side
            "word_count": 0.0,  # word counts for both source and target
            "sequence_count": 0.0,  # number of training examples processed
            "grad_norm": 0.0}


def before_train(loaded_train_model, train_model, train_sess, global_step, hparams, log_f):
    stats = init_stats()
    info = {'train_ppl': 0.0, 'speed': 0.0, 'avg_step_time': 0.0, 'avg_grad_norm': 0.0, 'avg_sequence_count': 0.0,
            'learning_rate': loaded_train_model.learning_rate.eval(session=train_sess)}

    skip_count = hparams.batch_size * hparams.epoch_step
    start_train_time = time.time()
    train_sess.run(loaded_train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: skip_count})
    return stats, info, start_train_time


def train(hparams, scope=None, target_session=''):
    steps_per_stats = hparams.steps_per_stats  # 输出统计量的间隔.

    out_dir = hparams.output_dir
    model_initializer = get_model_creator(hparams)
    train_model_tuple = create_train_model_tuple(model_initializer, hparams, scope)
    eval_model_tuple = create_eval_model_tuple(model_initializer,hparams,scope)

    config_proto = get_config_proto(log_device_placement=None, allow_soft_placement=True)
    summary_name = 'train_log'
    summary_writer = tf.summary.FileWriter(os.path.join(out_dir, summary_name), graph=train_model_tuple.graph)



    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model_tuple.graph)
    # train_sess =  tf_debug.LocalCLIDebugWrapperSession(tf.Session(target=target_session, config=config_proto, graph=train_model_tuple.graph))
    with train_model_tuple.graph.as_default():
        loaded_train_model, global_step = create_or_load_model(train_model_tuple.model, out_dir, train_sess, "train")


    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")

    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    stats, info, start_train_time = before_train(
        loaded_train_model, train_model_tuple, train_sess, global_step, hparams, log_f)

    while global_step < hparams.num_train_steps:
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError as e:
            hparams.epoch_step = 0
            print('Finished One Epoch with Global_Step %s' % global_step)
            train_sess.run(
                train_model_tuple.iterator.initializer,
                feed_dict={train_model_tuple.skip_count_placeholder: 0})
            continue

        global_step, info['learning_rate'], step_summary = update_stats(stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step  # 重置 last_stats_step 为 global_step.
            is_overflow = process_stats(stats, info, global_step, steps_per_stats, log_f)

            if is_overflow:
                break

            stats = init_stats()



def run_full_eval(model_dir,
                  infer_model,
                  infer_sess,
                  eval_model,
                  eval_sess,
                  hparams,
                  summary_writer,
                  sample_src_data,
                  sample_tgt_data,
                  avg_ckpts=False):
    pass


def run_internal_eval():
    pass

def run_external_eval():
    pass



def process_stats(stats, info, global_step, steps_per_stats, log_f):
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["avg_sequence_count"] = stats["sequence_count"] / steps_per_stats
    info["speed"] = stats["word_count"] / (1000 * stats["step_time"])

    info['train_ppl'] = safe_exp(stats['train_loss'] / stats['predict_count'])

    is_overflow = False

    train_ppl = info['train_ppl']

    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
        is_overflow = True

    return is_overflow


def update_stats(stats, start_time, step_result):
    _, output_tuple = step_result

    batch_size = output_tuple.batch_size
    stats['step_time'] += time.time() - start_time
    stats['train_loss'] += output_tuple.train_loss * batch_size
    stats['grad_norm'] += output_tuple.grad_norm
    stats['learning_rate'] = output_tuple.learning_rate

    return (output_tuple.global_step, output_tuple.learning_rate, output_tuple.train_summary)


def run_main(FLAGS, hparams, train_fn, inference_fn):
    train_fn(hparams)


def main(unused_argv):
    hparams = create_hparams(FLAGS)
    train_fn = train
    inference_fn = None
    run_main(FLAGS, hparams, train_fn, inference_fn)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    add_parameter(arg_parser)
    FLAGS, unparsed = arg_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
