import tensorflow as tf
import numpy as np
import os
from cnn.network.cnn_inct_para  import *

#======================================================================================================#

class cnn_inct_conf(object):

    def __init__(self, input_layer_val, mode):

        self.cnn_para        = cnn_inct_para()
        self.input_layer_val = input_layer_val
        self.mode            = mode
        
        ### ======== LAYER 01
        with tf.device('/gpu:0'), tf.variable_scope("cnn_conv01")as scope:
             [self.output_layer01, self.mid_layer01] = self.inct_layer(
                                                   self.input_layer_val,

                                                   self.cnn_para.l01_filter_height,
                                                   self.cnn_para.l01_filter_width,
                                                   self.cnn_para.l01_pre_filter_num,
                                                   self.cnn_para.l01_filter_num,
                                                   self.cnn_para.l01_conv_padding,
                                                   self.cnn_para.l01_conv_stride,

                                                   self.cnn_para.l01_is_norm,

                                                   self.cnn_para.l01_conv_act_func,

                                                   self.cnn_para.l01_is_pool,
                                                   self.cnn_para.l01_pool_type,
                                                   self.cnn_para.l01_pool_padding,
                                                   self.cnn_para.l01_pool_stride,
                                                   self.cnn_para.l01_pool_ksize,

                                                   self.cnn_para.l01_is_drop,
                                                   self.cnn_para.l01_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   
        ### ======== LAYER 02
        with tf.device('/gpu:0'), tf.variable_scope("cnn_conv02")as scope:
             [self.output_layer02, self.mid_layer02] = self.inct_layer(
                                                   self.output_layer01,

                                                   self.cnn_para.l02_filter_height,
                                                   self.cnn_para.l02_filter_width,
                                                   self.cnn_para.l02_pre_filter_num,
                                                   self.cnn_para.l02_filter_num,
                                                   self.cnn_para.l02_conv_padding,
                                                   self.cnn_para.l02_conv_stride,

                                                   self.cnn_para.l02_is_norm,

                                                   self.cnn_para.l02_conv_act_func,

                                                   self.cnn_para.l02_is_pool,
                                                   self.cnn_para.l02_pool_type,
                                                   self.cnn_para.l02_pool_padding,
                                                   self.cnn_para.l02_pool_stride,
                                                   self.cnn_para.l02_pool_ksize,

                                                   self.cnn_para.l02_is_drop,
                                                   self.cnn_para.l02_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   

        ### ======== LAYER 03
        with tf.device('/gpu:0'), tf.variable_scope("cnn_conv03")as scope:
             [self.output_layer03, self.mid_layer03] = self.inct_layer(
                                                   self.output_layer02,

                                                   self.cnn_para.l03_filter_height,
                                                   self.cnn_para.l03_filter_width,
                                                   self.cnn_para.l03_pre_filter_num,
                                                   self.cnn_para.l03_filter_num,
                                                   self.cnn_para.l03_conv_padding,
                                                   self.cnn_para.l03_conv_stride,

                                                   self.cnn_para.l03_is_norm,

                                                   self.cnn_para.l03_conv_act_func,

                                                   self.cnn_para.l03_is_pool,
                                                   self.cnn_para.l03_pool_type,
                                                   self.cnn_para.l03_pool_padding,
                                                   self.cnn_para.l03_pool_stride,
                                                   self.cnn_para.l03_pool_ksize,

                                                   self.cnn_para.l03_is_drop,
                                                   self.cnn_para.l03_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  )   

        ### ======== LAYER 04
        with tf.device('/gpu:0'), tf.variable_scope("cnn_conv04")as scope:
             [self.output_layer04, self.mid_layer04] = self.inct_layer(
                                                   self.output_layer03,

                                                   self.cnn_para.l04_filter_height,
                                                   self.cnn_para.l04_filter_width,
                                                   self.cnn_para.l04_pre_filter_num,
                                                   self.cnn_para.l04_filter_num,
                                                   self.cnn_para.l04_conv_padding,
                                                   self.cnn_para.l04_conv_stride,

                                                   self.cnn_para.l04_is_norm,

                                                   self.cnn_para.l04_conv_act_func,

                                                   self.cnn_para.l04_is_pool,
                                                   self.cnn_para.l04_pool_type,
                                                   self.cnn_para.l04_pool_padding,
                                                   self.cnn_para.l04_pool_stride,
                                                   self.cnn_para.l04_pool_ksize,

                                                   self.cnn_para.l04_is_drop,
                                                   self.cnn_para.l04_drop_prob,

                                                   self.mode,
                                                   scope=scope
                                                  ) 
             self.final_output = self.output_layer04

###==================================================== OTHER FUNCTION ============================
    #02/ CONV LAYER
    def inct_layer(self, 
                  input_value, 

                  filter_height, 
                  filter_width, 
                  pre_filter_num, 
                  filter_num, 
                  conv_padding, 
                  conv_stride,

                  is_norm,

                  act_func,

                  is_pool, 
                  pool_type, 
                  pool_padding, 
                  pool_stride, 
                  pool_ksize, 

                  is_drop,
                  drop_prob,

                  mode,
                  scope=None
                 ):
        #------------------------------#
        def reduce_var(x, axis=None, keepdims=False, name=None):
            m = tf.reduce_mean(x, axis=axis, keepdims=True, name=name) #keep same dimension for subtraction
            devs_squared = tf.square(x - m)
            return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims, name=name)
        
        def reduce_std(x, axis=None, keepdims=False, name=None):
            return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims, name=name))

        #------------------------------#

        with tf.variable_scope(scope or 'conv-layer') as scope:
            #BachNorm Layer
            if(is_norm == True):
                batch_output_01 = tf.contrib.layers.batch_norm(input_value, 
                                                            is_training = mode, 
                                                            decay = 0.9,
                                                            zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output_01 = input_value

            #============ inception layer herer
            # 1x1
            filter_height_01 = 1
            filter_width_01  = 1
            filter_num_01    = filter_num/8
            filter_shape_01  = [int(filter_height_01), int(filter_width_01), int(pre_filter_num), int(filter_num_01)]
            W_01             = tf.Variable(tf.truncated_normal(filter_shape_01, stddev=0.1), name="W_01")   # this is kernel 
            b_01             = tf.Variable(tf.constant(0.1, shape=[int(filter_num_01)]), name="b_01")
            conv_output_01   = tf.nn.conv2d(batch_output_01,
                                            W_01,
                                            strides = conv_stride,
                                            padding = conv_padding,
                                            name="conv_01"
                                           )  
            conv_output_01 = tf.nn.bias_add(conv_output_01, b_01)

            # 3x3
            filter_height_02 = 3
            filter_width_02  = 3
            filter_num_02    = filter_num/2
            filter_shape_02  = [int(filter_height_02), int(filter_width_02), int(pre_filter_num), int(filter_num_02)]
            W_02             = tf.Variable(tf.truncated_normal(filter_shape_02, stddev=0.1), name="W_02")   # this is kernel 
            b_02             = tf.Variable(tf.constant(0.1, shape=[int(filter_num_02)]), name="b_02")
            conv_output_02   = tf.nn.conv2d(batch_output_01,
                                            W_02,
                                            strides = conv_stride,
                                            padding = conv_padding,
                                            name="conv_02"
                                           )  #default: data format = NHWC
            conv_output_02 = tf.nn.bias_add(conv_output_02, b_02)

            # 5x5 - learn time dim
            filter_height_03 = 1
            filter_width_03  = 4
            filter_num_03    = filter_num/4
            filter_shape_03  = [int(filter_height_03), int(filter_width_03), int(pre_filter_num), int(filter_num_03)]
            W_03             = tf.Variable(tf.truncated_normal(filter_shape_03, stddev=0.1), name="W_03")   # this is kernel 
            b_03             = tf.Variable(tf.constant(0.1, shape=[int(filter_num_03)]), name="b_03")
            conv_output_03   = tf.nn.conv2d(batch_output_01,
                                            W_03,
                                            strides = conv_stride,
                                            padding = conv_padding,
                                            name="conv_03"
                                           )  #default: data format = NHWC
            conv_output_03 = tf.nn.bias_add(conv_output_03, b_03)


            # pool 05
            pool_output_04 = tf.nn.max_pool(batch_output_01,
                                            ksize   = [1,3,3,1],   
                                            strides = conv_stride,
                                            padding = conv_padding,
                                            name="pool_04"
                                            )
            # 1x1 - of pool
            filter_height_04 = 1
            filter_width_04  = 1
            filter_num_04    = filter_num/8
            filter_shape_04  = [int(filter_height_04), int(filter_width_04), int(pre_filter_num), int(filter_num_04)]
            W_04             = tf.Variable(tf.truncated_normal(filter_shape_04, stddev=0.1), name="W_04")   # this is kernel 
            b_04             = tf.Variable(tf.constant(0.1, shape=[int(filter_num_04)]), name="b_04")
            conv_output_04   = tf.nn.conv2d(pool_output_04,
                                            W_04,
                                            strides = conv_stride,
                                            padding = conv_padding,
                                            name="conv_04"
                                           )  #default: data format = NHWC
            conv_output_04 = tf.nn.bias_add(conv_output_04, b_04)

            # concat
            conv_output = tf.concat((conv_output_01, conv_output_02, conv_output_03, conv_output_04),3)
            #print(conv_output.get_shape())
            #exit()

            #Active function layer
            if (act_func == 'RELU'):
                act_func_output = tf.nn.elu(conv_output, name="RELU")
            elif (act_func == 'TANH'):
                act_func_output = tf.nn.tanh(conv_output, name="TANH")

            #BachNorm Layer
            if(is_norm == True):
                batch_output = tf.contrib.layers.batch_norm(
                                                             act_func_output, 
                                                             is_training = mode, 
                                                             decay = 0.9,
                                                             zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output = act_func_output

            #Pooling layer
            if(is_pool == True):
                if (pool_type == 'MEAN'):
                    pool_output = tf.nn.avg_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="mean_pool"
                                         )
                elif (pool_type == 'MAX'):
                    pool_output = tf.nn.max_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="max_pool"
                                         )
                elif (pool_type == 'GLOBAL_MAX'):
                    pool_output = tf.reduce_max(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_max'
                                         )
                elif (pool_type == 'GLOBAL_MEAN'):
                    pool_output = tf.reduce_mean(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_moment01_pool'
                                         )
                elif (pool_type == 'GLOBAL_STD'):   #only for testing (not apply for training)
                    pool_output = reduce_std(
                                          batch_output,
                                          axis=[1,2],
                                          name = "global_moment02_pool"
                                         )
                    #print pool_output.get_shape()
                    #exit()
            else:
                pool_output = batch_output

            #Dropout
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                pool_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:     
                drop_output = pool_output

            mid_output = conv_output
            return drop_output, mid_output

    #02/ CONV LAYER
    def conv_layer(self, 
                  input_value, 

                  filter_height, 
                  filter_width, 
                  pre_filter_num, 
                  filter_num, 
                  conv_padding, 
                  conv_stride,

                  is_norm,

                  act_func,

                  is_pool, 
                  pool_type, 
                  pool_padding, 
                  pool_stride, 
                  pool_ksize, 

                  is_drop,
                  drop_prob,

                  mode,
                  scope=None
                 ):
        #------------------------------#
        def reduce_var(x, axis=None, keepdims=False, name=None):
            m = tf.reduce_mean(x, axis=axis, keepdims=True, name=name) #keep same dimension for subtraction
            devs_squared = tf.square(x - m)
            return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims, name=name)
        
        def reduce_std(x, axis=None, keepdims=False, name=None):
            return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims, name=name))

        #------------------------------#

        with tf.variable_scope(scope or 'conv-layer') as scope:
            #BachNorm Layer
            if(is_norm == True):
                batch_output_01 = tf.contrib.layers.batch_norm(input_value, 
                                                            is_training = mode, 
                                                            decay = 0.9,
                                                            zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output_01 = input_value

            # shape: [5,5,1,32] or [5,5,32,64]
            filter_shape = [filter_height, filter_width, pre_filter_num, filter_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")   # this is kernel 
            b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")

            #Convolution layer 
            conv_output = tf.nn.conv2d(
                                 batch_output_01,
                                 W,
                                 strides = conv_stride,
                                 padding = conv_padding,
                                 name="conv"
                                 )  #default: data format = NHWC


            #Active function layer
            if (act_func == 'RELU'):
                act_func_output = tf.nn.elu(tf.nn.bias_add(conv_output, b), name="RELU")
            elif (act_func == 'TANH'):
                act_func_output = tf.nn.tanh(tf.nn.bias_add(conv_output, b), name="TANH")

            #BachNorm Layer
            if(is_norm == True):
                batch_output = tf.contrib.layers.batch_norm(
                                                             act_func_output, 
                                                             is_training = mode, 
                                                             decay = 0.9,
                                                             zero_debias_moving_mean=True
                                                           )
            else:     
                batch_output = act_func_output

            #Pooling layer
            if(is_pool == True):
                if (pool_type == 'MEAN'):
                    pool_output = tf.nn.avg_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="mean_pool"
                                         )
                elif (pool_type == 'MAX'):
                    pool_output = tf.nn.max_pool(
                                          batch_output,
                                          ksize   = pool_ksize,   
                                          strides = pool_stride,
                                          padding = pool_padding,
                                          name="max_pool"
                                         )
                elif (pool_type == 'GLOBAL_MAX'):
                    pool_output = tf.reduce_max(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_max'
                                         )
                elif (pool_type == 'GLOBAL_MEAN'):
                    pool_output = tf.reduce_mean(
                                          batch_output,
                                          axis=[1,2],
                                          name='global_moment01_pool'
                                         )
                elif (pool_type == 'GLOBAL_STD'):   #only for testing (not apply for training)
                    pool_output = reduce_std(
                                          batch_output,
                                          axis=[1,2],
                                          name = "global_moment02_pool"
                                         )
                    #print pool_output.get_shape()
                    #exit()
            else:
                pool_output = batch_output

            #Dropout
            if(is_drop == True):
                drop_output = tf.layers.dropout(
                                                pool_output, 
                                                rate = drop_prob,
                                                training = mode,
                                                name = 'Dropout'
                                               )
            else:     
                drop_output = pool_output

            mid_output = conv_output
            return drop_output, mid_output
