import numpy as np
import os


class rnn_para(object):

    def __init__(self):

        self.input_drop   = 0.7
        self.output_drop  = 0.7
        self.n_layer      = 1
        self.n_hidden     = 64
        self.nframe       = 64

