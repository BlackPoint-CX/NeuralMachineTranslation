#!/usr/bin/env bash

readonly python=/Users/alfredchen/Develop/VirtualEnvs/virtualenv_3.6.5/bin/python
readonly src_dir=/Users/alfredchen/PycharmProjects/NeuralMachineTranslation/nmt/src
readonly data_dir=/Users/alfredchen/PycharmProjects/NeuralMachineTranslation/nmt/data


readonly src_file=${data_dir}/iwslt15.tst2013.100.en
readonly tgt_file=${data_dir}/iwslt15.tst2013.100.vi

readonly src_vocab_file=${data_dir}/iwslt15.vocab.100.en
readonly tgt_vocab_file=${data_dir}/iwslt15.vocab.100.vi

readonly output_dir=${src_dir}/output

#readonly src_embed_file=${data_dir}/iwslt15.vocab.100.en
#readonly tgt_embed_file=${data_dir}/iwslt15.vocab.100.vi


echo "Python Version"
${python} --version

echo "Process training"
${python} ${src_dir}/entrance.py    --num_units 32  \
                                    --src_vocab_file ${src_vocab_file} \
                                    --tgt_vocab_file ${tgt_vocab_file} \
                                    --src_file ${src_file} \
                                    --tgt_file ${tgt_file} \
                                    --dropout 0.2 \
                                    --mode train \
                                    --num_encoder_layers 2 \
                                    --num_decoder_layers 2 \
                                    --output_dir ${output_dir}



#parser.add_argument('--num_units', type=int, default=32, help='Network size.')
#parser.add_argument('--src_vocab_file', type=str, default='', help='Path of source vocabulary file.')
#parser.add_argument('--tgt_vocab_file', type=str, default='', help='Path of target vocabulary file.')
#parser.add_argument('--src_embed_file', type=str, default='', help='Path of source embedding file.')
#parser.add_argument('--tgt_embed_file', type=str, default='', help='Path of target embedding file.')
#parser.add_argument('--src_vocab_size', type=int, default=None, help='Size of source vocabulary.')
#parser.add_argument('--tgt_vocab_size', type=int, default=None, help='Size of target vocabulary.')
#parser.add_argument('--num_encoder_layers', type=int, default=1, help='Num of layers in encoder.')
#parser.add_argument('--num_decoder_layers', type=int, default=1, help='Num of layers in decoder.')
#parser.add_argument('--mode', type=str, default='train', help='Mode : train | eval | infer')
#parser.add_argument('--dropout', type=float, default=None, help='Droupout rate.')
#parser.add_argument('--src_file', type=str, default='', help='Source file for training.')
#parser.add_argument('--tgt_file', type=str, default='', help='Target file for training.')
