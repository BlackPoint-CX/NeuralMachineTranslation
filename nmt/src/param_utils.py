from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.training import HParams


def add_parameter(parser):
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--num_units', type=int, default=32, help='Network size.')
    parser.add_argument('--src_vocab_file', type=str, default='', help='Path of source vocabulary file.')
    parser.add_argument('--tgt_vocab_file', type=str, default='', help='Path of target vocabulary file.')
    parser.add_argument('--src_embed_file', type=str, default='', help='Path of source embedding file.')
    parser.add_argument('--tgt_embed_file', type=str, default='', help='Path of target embedding file.')
    parser.add_argument('--src_vocab_size', type=int, default=100, help='Size of source vocabulary.')
    parser.add_argument('--tgt_vocab_size', type=int, default=100, help='Size of target vocabulary.')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Num of layers in encoder.')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Num of layers in decoder.')
    parser.add_argument('--mode', type=str, default='train', help='Mode : train | eval | infer')
    parser.add_argument('--dropout', type=float, default=None, help='Droupout rate.')
    parser.add_argument('--src_file', type=str, default='', help='Source file for training.')
    parser.add_argument('--tgt_file', type=str, default='', help='Target file for training.')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory.')
    parser.add_argument('-num_keep_ckpts', type=int, default=5, help='Maximum number of saved model files.')
    parser.add_argument('--time_major', type='bool', default=False, help='Time major or not.')

    parser.add_argument('--encoder_type', type=str, default='uni', help='Direction of RNN cell.')
    parser.add_argument('--sos', type=str, default='<s>', help='Start of sentence symbol.')
    parser.add_argument('--eos', type=str, default='</s>', help='End of sentence symbol.')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd | adam')
    parser.add_argument('--src_max_len', type=int, default=50, help='Max lengths of src sequences during training.')
    parser.add_argument('--tgt_max_len', type=int, default=50, help='Max lengths of tgt sequences during training.')
    parser.add_argument('--infer_mode', type=str, default='greedy', choices=['greedy', 'sample', 'beam_search'])
    parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='Clip gradients to this norm.')
    parser.add_argument('--model_type', type=str, default='standard', help='Choice of model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--unit_type', type=str, default='lstm', help='Type of rnn cell')
    parser.add_argument('--forget_bias', type=float, default=1.0, help='Forget bias.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of layers.')
    parser.add_argument('--num_train_steps', type=int, default=1000, help='Num of train steps.')


def create_hparams(FLAGS):
    return HParams(
        num_units=FLAGS.num_units,
        src_vocab_file=FLAGS.src_vocab_file,
        tgt_vocab_file=FLAGS.tgt_vocab_file,
        src_embed_file=FLAGS.src_embed_file,
        tgt_embed_file=FLAGS.tgt_embed_file,
        src_vocab_size=FLAGS.src_vocab_size,
        tgt_vocab_size=FLAGS.tgt_vocab_size,
        num_encoder_layers=FLAGS.num_encoder_layers,
        num_decoder_layers=FLAGS.num_decoder_layers,
        mode=FLAGS.mode,
        dropout=FLAGS.dropout,
        src_file=FLAGS.src_file,
        tgt_file=FLAGS.tgt_file,
        output_dir=FLAGS.output_dir,
        num_keep_ckpts=FLAGS.num_keep_ckpts,
        time_major=FLAGS.time_major,
        encoder_type=FLAGS.encoder_type,
        sos=FLAGS.sos,
        eos=FLAGS.eos,
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        src_max_len=FLAGS.src_max_len,
        tgt_max_len=FLAGS.tgt_max_len,
        infer_mode=FLAGS.infer_mode,
        max_gradient_norm=FLAGS.max_gradient_norm,
        model_type=FLAGS.model_type,
        batch_size=FLAGS.batch_size,
        unit_type=FLAGS.unit_type,
        forget_bias=FLAGS.forget_bias,
        num_layers=FLAGS.num_layers,
        epoch_step=0,
        num_train_steps=FLAGS.num_train_steps

    )
