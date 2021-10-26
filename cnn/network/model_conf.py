import tensorflow as tf
import numpy as np
import os

from model_para import *

#from cnn_bl_time_conf import *
#from cnn_bl_freq_conf import *
from cnn_bl_conf import *
#from cnn_inct_conf import *

#from dnn_bl01_conf import *
from dnn_bl02_conf import *

from nn_basic_layers import *
from rnn_para import *

#======================================================================================================#

class model_conf(object):

    def __init__( self):

        self.model_para = model_para()
        self.rnn_para   = rnn_para()

        # ============================== Fed Input
        self.input_layer_val     = tf.placeholder(tf.float32, [None, self.model_para.n_freq, self.model_para.n_time, self.model_para.n_chan], name="input_layer_val")


        self.expected_classes    = tf.placeholder(tf.float32, [None, self.model_para.n_class], name="expected_classes")
        self.mode                = tf.placeholder(tf.bool, name="running_mode")

        self.seq_len             = tf.placeholder(tf.int32, [None], name="seq_len" ) # for the dynamic RNN

        #============================== NETWORK CONFIGURATION

        # =================== 01/ C-RNN

        # Call CNN Time and Get CNN Time output
        with tf.device('/gpu:0'), tf.variable_scope("cnn_time")as scope:
            self.cnn_ins_01 = cnn_bl_conf(self.input_layer_val, self.mode)
            self.merge_data = self.cnn_ins_01.final_output

        # Call DNN 
        with tf.device('/gpu:0'), tf.variable_scope("dnn_01")as scope:
            self.dnn_bl01_ins_01 = dnn_bl02_conf(self.merge_data, 512, self.mode)

        with tf.device('/gpu:0'), tf.variable_scope("output")as scope:
            # ======================================  Output =================================== #
            self.output_layer      = self.dnn_bl01_ins_01.final_output
            self.prob_output_layer = tf.nn.softmax(self.output_layer)
            self.wanted_data       = self.merge_data 

        ### ======================================== LOSS FUNCTION AND ACCURACY =========================
        ### loss function
        with tf.device('/gpu:0'), tf.variable_scope("loss") as scope:

            # l2 loss
            l2_loss = self.model_para.l2_lamda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

            # main loss
            #losses     = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.expected_classes, logits=self.output_layer)
            dummy = tf.constant(0.00001) # to avoid dividing by 0 with KL divergence
            p = self.expected_classes  + dummy
            q = self.prob_output_layer + dummy
            losses = tf.reduce_sum(p * tf.log(p/q))

            # final loss
            #self.loss = (tf.reduce_mean(losses_b1) + tf.reduce_mean(losses_b2) )/2 + tf.reduce_mean(losses)
            self.loss = losses+l2_loss

        ### Calculate Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy") as scope:
            self.correct_prediction    = tf.equal(tf.argmax(self.output_layer,1),    tf.argmax(self.expected_classes,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,"float", name="accuracy" ))
