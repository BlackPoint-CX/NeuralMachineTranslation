from configparser import ConfigParser


class NMTConfig(object):
    def __init__(self, args):
        self.args = args
        self.num_units = None  # Number of units in encoder layer
        self.batch_size = None
        # ID of starting
        self.tgt_sos_id = None
        # ID of ending
        self.tgt_eos_id = None
        self.tgt_vocab_size = None

        self.args.inference_input_file = None
        self.args.inference_output_file = None
        self.source_word_vocab_size = None
        self.source_word_embedding_dim = None
        self.source_pretrain_embedding = None
        self.use_pretrain_embedding = None
        self.tgt_vocab_size = None
