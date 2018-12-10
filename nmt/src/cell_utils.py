import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, NASCell
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, DropoutWrapper, GRUCell, MultiRNNCell


def build_single_cell(mode, unit_type, num_units, forget_bias, dropout):
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


def build_cell_list(num_layers, mode, unit_type, num_units, forget_bias, dropout, single_cell_fn=None):
    if not single_cell_fn:
        single_cell_fn = build_single_cell

    cell_list = []

    for i in range(num_layers):
        single_cell = single_cell_fn(mode=mode,
                                     unit_type=unit_type,
                                     num_units=num_units,
                                     forget_bias=forget_bias,
                                     dropout=dropout)
        cell_list.append(single_cell)

    return cell_list


def build_rnn_cell(num_layers, mode, unit_type, num_units, forget_bias, dropout, single_cell_fn=None):
    cell_list = build_cell_list(num_layers, mode, unit_type, num_units, forget_bias, dropout,
                                single_cell_fn=None)
    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return MultiRNNCell(cells=cell_list)


def build_encoder_cell(hparmas):
    return build_rnn_cell(num_layers=hparmas.num_layers,
                          mode=hparmas.mode,
                          unit_type=hparmas.unit_type,
                          num_units=hparmas.num_units,
                          forget_bias=hparmas.forget_bias,
                          dropout=hparmas.dropout,
                          single_cell_fn=hparmas.single_cell_fn
                          )


def build_decoder_cell(self):
    pass


def build_bidirectional_rnn(self, inputs, sequence_length, dtype, hparams, num_bi_layers=None,
                            num_bi_residual_layers=None):
    fw_cell = build_encoder_cell(hparams)
    bw_cell = build_encoder_cell(hparams)
    bi_output, bi_state = bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs,
                                                    sequence_length=sequence_length)
    encoder_outputs = tf.concat(bi_output, -1)
    encoder_state = bi_state
    return encoder_outputs, encoder_state
