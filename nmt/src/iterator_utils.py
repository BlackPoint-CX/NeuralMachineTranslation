from collections import namedtuple


class BatchedInput(namedtuple(typename='BatchedInput',
                              field_names=('initializer', 'source', 'target_input', 'target_output',
                                           'source_sequence_length', 'target_sequence_length'))):
    pass
