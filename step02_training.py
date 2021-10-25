import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf
import argparse
import math
import scipy.io
#from scipy.io import loadmat
import re
import time
import datetime
import sys
from sklearn import datasets, svm, metrics
from numpy import random

sys.path.append('./network/')
from model_conf import *

#import data_helpers
#from shutil import copyfile
#import h5py


#=========================================== 01/ PARAMETERS
random.seed(1)
print("\n ==================================================================== SETUP PARAMETERS...")

# 1.1/ Directory TODO
tf.flags.DEFINE_string("TRAIN_DIR",  "./../../data/data_train/", "Directory of feature")
tf.flags.DEFINE_string("VALID_DIR",  "./../../data/data_test/", "Directory of feature")


tf.flags.DEFINE_string("OUT_DIR",    "./data/",     "Point to output directory")

# 1.2/ Training para TODO
tf.flags.DEFINE_integer("N_TRAIN_MUL_BATCH", 21,     "Multi Batch Number for Training")   #106-157
tf.flags.DEFINE_integer("BATCH_SIZE",        50,    "Batch Size ")
tf.flags.DEFINE_integer("NUM_EPOCHS",        100,    "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("N_CLASS",           4,     "Class Number")
tf.flags.DEFINE_integer("N_VALID",           2756,   "Valid file number") #2878

tf.flags.DEFINE_integer("CHECKPOINT_EVERY",  20,   "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float  ("LEARNING_RATE",     1e-4,  "Learning rate")
tf.flags.DEFINE_integer("N_FRAME",           64,    "time/freq resolution")     #This is for RNN --> need to matching rnn_para.py

# 1.3/ Device Report Para
tf.flags.DEFINE_boolean("allow_soft_placement", True,  "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
mixup_num = int(FLAGS.BATCH_SIZE/2)

#======================================================  02/ HANDLE FILE
### train dir
train_dir = os.path.abspath(FLAGS.TRAIN_DIR)
org_train_file_list = os.listdir(train_dir)
train_file_list = []  #remove .file
for nFileTrain in range(0,len(org_train_file_list)):
    isHidden=re.match("\.",org_train_file_list[nFileTrain])
    if (isHidden is None):
        train_file_list.append(org_train_file_list[nFileTrain])
train_file_list = sorted(train_file_list)        

### valid dir
valid_dir = os.path.abspath(FLAGS.VALID_DIR)
org_valid_file_list = os.listdir(valid_dir)
valid_file_list = []  #remove .file
for nClassValid in range(0,len(org_valid_file_list)):
    isHidden=re.match("\.",org_valid_file_list[nClassValid])
    if (isHidden is None):
        valid_file_list.append(org_valid_file_list[nClassValid])
valid_file_num  = len(valid_file_list)
valid_file_list = sorted(valid_file_list)


#======================================================  03/ TRAINING & SAVE
print("\n ==================================================================== TRAINING DATA...")

tf.reset_default_graph()
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=False)
    tf.set_random_seed(1)
    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement, 
                                   log_device_placement=FLAGS.log_device_placement
                                   #gpu_options = gpu_options
                                 )

    sess = tf.Session(config=session_conf)

    with sess.as_default():

        # ==================================================  01/ instance network model
        print("\n =============== 01/ Instance Model")
        model = model_conf()


        # 02/ Define Training procedure, optional optimizer ....
        print("\n =============== 02/ Setting Training Options")
        print("\n + Adam optimizer ")
        print("\n + Learning Rate: {}".format(FLAGS.LEARNING_RATE))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step    = tf.Variable(0, name="global_step", trainable=False)
            optimizer      = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op       = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # ====================================================  02/ Setup kinds of report summary 
        print("\n =============== 03/ Setting Report ...")
        print("\n + Gradient ")
        print("\n + Sparsity ")
        print("\n + Loss ")
        print("\n + Accuracy ")

        ### TODO-Report#   02/01 Gradient and Sparsity 
        ### grad_summaries = []
        ### for g, v in grads_and_vars:
        ###     if g is not None:
        ###         grad_hist_summary = tf.summary.histogram ("{}/grad/hist".format(v.name), g)
        ###         sparsity_summary  = tf.summary.scalar    ("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        ###         grad_summaries.append(grad_hist_summary)
        ###         grad_summaries.append(sparsity_summary)
        ### grad_summaries_merged = tf.summary.merge(grad_summaries)

        ### #   02/02 loss and accuracy 
        ### loss_summary     = tf.summary.scalar("loss", model.loss)
        ### acc_summary      = tf.summary.scalar("accuracy", model.accuracy)
        ### 
        ### #   02/03 Merge all reported parameters
        ### #train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        ### train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])


        # ====================================================   03/ Setup training summary directory
        print("\n =============== 04/ Setting Directory for Saving...")
        stored_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.OUT_DIR))
        print("+ Writing to {}\n".format(stored_dir))

        train_summary_dir = os.path.join(stored_dir, "summaries", "train")   #stored_dir/summaries/train
        print("+ Training summary Writing to {}\n".format(train_summary_dir))

        ### TODO-Report#register direction to add summaries
        ### train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(stored_dir, "checkpoints"))
        print("XXXXXXXXXXXXXXXXX: Checkpoint Dir: {}\n".format(checkpoint_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_model_dir = os.path.join(stored_dir, "model")
        print("XXXXXXXXXXXXXXXXX: Best model Dir: {}\n".format(best_model_dir))
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)


        ### ======================================================= 04/ Save and initial/load best model
        # Create saver     
        print("\n =============== 05/ Creating Saver...")
        saver = tf.train.Saver(tf.global_variables())

        # Load saved model to continue training or initialize all variables for new Model
        best_model_files     = os.path.join(best_model_dir, "best_model")
        best_model_meta_file = os.path.join(best_model_dir, "best_model.meta")
        print("XXXXXXXXXXXXXXXXX: Best Model Files: {}\n".format(best_model_files))
        print("XXXXXXXXXXXXXXXXX: Best Model Meta File: {}\n".format(best_model_meta_file))

        if os.path.isfile(best_model_meta_file):
            print("\n=============== 06/ Latest Model Loaded from dir: {}" .format(best_model_dir))
            saver = tf.train.import_meta_graph(best_model_meta_file)
            saver.restore(sess, best_model_files)
        else:
            print("\n=============== 06/ New Model Initialized")
            sess.run(tf.global_variables_initializer())

        # ============================================================ 05/ Define training function that is called every epoch
        def train_process(x_train_batch, y_train_batch):
            # Training every batch
            [nS, nF, nT, nC] = x_train_batch.shape
            seq_len = np.ones(int(nS),dtype=int)*FLAGS.N_FRAME 

            feed_dict= {model.input_layer_val:   x_train_batch,
                        model.expected_classes:  y_train_batch,
                        model.seq_len: seq_len,
                        model.mode: True
                       }

            # Training and return data
            #[ _, step, summaries, loss, accuracy] = sess.run([train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
            #[ _, step, summaries, loss, accuracy, end_output] = sess.run([train_op, global_step, train_summary_op, model.loss, model.accuracy, model.output_layer], feed_dict)
            # Remove summary report
            [ _, step, loss, accuracy, end_output] = sess.run([train_op, global_step, model.loss, model.accuracy, model.output_layer], feed_dict)

            #time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))    #pint here --> no need returna

            ### TODO-Report add summaries every epoch
            ### train_summary_writer.add_summary(summaries, step)

            return accuracy, end_output

        def test_one_batch(x_test_batch, y_test_batch):
            [nS, nF, nT, nC] = x_test_batch.shape
            seq_len = np.ones(int(nS),dtype=int)*FLAGS.N_FRAME 

            feed_dict= {model.input_layer_val:   x_test_batch,
                        model.expected_classes:  y_test_batch,
                        model.seq_len: seq_len,
                        model.mode: False
                       }

            [loss, accuracy] = sess.run([model.loss, model.accuracy], feed_dict)

            return accuracy

        def test_process(x_test_batch):
            # Training every batch
            [nS, nF, nT, nC] = x_test_batch.shape
            seq_len = np.ones(int(nS),dtype=int)*FLAGS.N_FRAME 

            feed_dict= {model.input_layer_val:  x_test_batch,
                        model.seq_len: seq_len,
                        model.mode: False
                       }

            # Training and return data
            [step, end_output] = sess.run([global_step, model.prob_output_layer], feed_dict)

            #time_str = datetime.datetime.now().isoformat()
            #print("TESTING_AT {} and step {}: loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))   
            return end_output

        def get_train_batch(nTrainMulBatch):
            #print ("========== Obtain train data from the multibatch : {:g}".format(nTrainMulBatch + 1))

            train_file = train_dir + '/' + train_file_list[nTrainMulBatch]
            data_train        = np.load(train_file)     
            x_train_mul_batch = data_train['seq_x']
            y_train_mul_batch = data_train['seq_y']

            [nS, nF, nT] = x_train_mul_batch.shape
            x_train_mul_batch = np.reshape(x_train_mul_batch, [nS,nF,nT,1])      
            train_mul_batch_num = nS

            return x_train_mul_batch, y_train_mul_batch, train_mul_batch_num
        
        def get_test_batch(test_file_dir):

            data_test    = np.load(test_file_dir)     
            x_test_batch = data_test['seq_x']
  
            [nS, nF, nT] = x_test_batch.shape
            x_test_batch = np.reshape(x_test_batch, [nS,nF,nT,1])      

            return x_test_batch

        ### ============================================================  06/ Call epoch, train, validate, and test
        is_training   = 1
        is_validating = 1
        is_testing    = 1

        start_multi_batch = 1
        old_ave_acc       = 0

        is_break = 0
        if(is_training):
            ### Every Epoch
            for nEpoch in range(FLAGS.NUM_EPOCHS):
                if (is_break == 1):
                    break
                print("\n=======================  Epoch is", nEpoch, ";============================")
                #Every multiple batch
                for nTrainMulBatch in range(start_multi_batch, int(FLAGS.N_TRAIN_MUL_BATCH)):
                    if (is_break == 1):
                        break
                    # get data for every batch
                    [x_train_mul_batch, y_train_mul_batch, train_mul_batch_num] = get_train_batch(nTrainMulBatch)

                    #Every batch  in multi-batches
                    for nBatch in range(int(train_mul_batch_num/FLAGS.BATCH_SIZE)):
                        if (is_break == 1):
                            break

                        stPt = nBatch*FLAGS.BATCH_SIZE
                        edPt = (nBatch+1)*FLAGS.BATCH_SIZE 

                        x_train_batch = x_train_mul_batch[stPt:edPt, :, :, :]
                        y_train_batch = y_train_mul_batch[stPt:edPt, :]

                        # Mixture here 
                        X1      = x_train_batch[:mixup_num]
                        X2      = x_train_batch[mixup_num:]
                        y1      = y_train_batch[:mixup_num]
                        y2      = y_train_batch[mixup_num:]

                        # Betal dis
                        b   = np.random.beta(0.4, 0.4, mixup_num)
                        X_b = b.reshape(mixup_num, 1, 1, 1)
                        y_b = b.reshape(mixup_num, 1)

                        xb_mix   = X1*X_b     + X2*(1-X_b)
                        xb_mix_2 = X1*(1-X_b) + X2*X_b
                        yb_mix   = y1*y_b     + y2*(1-y_b)
                        yb_mix_2 = y1*(1-y_b) + y2*y_b
     
                        ## Uniform dis
                        #l   = np.random.random(mixup_num)
                        #X_l = l.reshape(mixup_num, 1, 1, 1)
                        #y_l = l.reshape(mixup_num, 1)

                        #xl_mix   = X1*X_l     + X2*(1-X_l)
                        #xl_mix_2 = X1*(1-X_l) + X2*X_l
                        #yl_mix   = y1* y_l    + y2 * (1-y_l)
                        #yl_mix_2 = y1*(1-y_l) + y2*y_l

                        #x_train_batch = np.concatenate((xb_mix, X1, xl_mix, xb_mix_2, X2, xl_mix_2), 0)   
                        #y_train_batch = np.concatenate((yb_mix, y1, yl_mix, yb_mix_2, y2, yl_mix_2), 0)
                        x_train_batch = np.concatenate((xb_mix, X1, xb_mix_2, X2), 0)   
                        y_train_batch = np.concatenate((yb_mix, y1, yb_mix_2, y2), 0)
                        #x_train_batch = np.concatenate((xb_mix, xl_mix, xb_mix_2, xl_mix_2), 0)   
                        #y_train_batch = np.concatenate((yb_mix, yl_mix, yb_mix_2, yl_mix_2), 0)
                        #print (np.shape(x_train_batch)) #300x378x128x1
                        #exit()
                        
                        # Call training process
                        train_acc, train_end_output = train_process(x_train_batch, y_train_batch)

                        # At check poit, verify the accuracy of testing set, and update the best model due to accuracy
                        current_step = tf.train.global_step(sess, global_step)
                        if (current_step % FLAGS.CHECKPOINT_EVERY == 0):  
                            print("Total Data Training Accuracy At Step {}: {}".format(current_step, train_acc))
                            with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                text_file.write("{0}\n".format(train_acc))

                            #01/ Save model when success testing on the first multi batch
                            if(is_testing):
                                #nMulBatchTest = random.randint(0, 75)
                                nMulBatchTest = 0
                                [x_test_mul_batch, y_test_mul_batch, test_mul_batch_num] = get_train_batch(nMulBatchTest)

                                acc_mul_batch_0 = 0
                                for nTestBatch in range(int(test_mul_batch_num/FLAGS.BATCH_SIZE)):

                                    stPt = nTestBatch*FLAGS.BATCH_SIZE
                                    edPt = (nTestBatch+1)*FLAGS.BATCH_SIZE 

                                    x_test_batch = x_test_mul_batch[stPt:edPt, :, :, :]
                                    y_test_batch = y_test_mul_batch[stPt:edPt, :]

                                    ## Mixture here 
                                    #X1      = x_test_batch[:mixup_num]
                                    #X2      = x_test_batch[mixup_num:]
                                    #y1      = y_test_batch[:mixup_num]
                                    #y2      = y_test_batch[mixup_num:]
            
                                    ## Betal dis
                                    #b   = np.random.beta(0.4, 0.4, mixup_num)
                                    #X_b = b.reshape(mixup_num, 1, 1, 1)
                                    #y_b = b.reshape(mixup_num, 1)
            
                                    #xb_mix   = X1*X_b     + X2*(1-X_b)
                                    #xb_mix_2 = X1*(1-X_b) + X2*X_b
                                    #yb_mix   = y1*y_b     + y2*(1-y_b)
                                    #yb_mix_2 = y1*(1-y_b) + y2*y_b
                 
                                    ## Uniform dis
                                    #l   = np.random.random(mixup_num)
                                    #X_l = l.reshape(mixup_num, 1, 1, 1)
                                    #y_l = l.reshape(mixup_num, 1)
            
                                    #xl_mix   = X1*X_l     + X2*(1-X_l)
                                    #xl_mix_2 = X1*(1-X_l) + X2*X_l
                                    #yl_mix   = y1* y_l    + y2 * (1-y_l)
                                    #yl_mix_2 = y1*(1-y_l) + y2*y_l
            
                                    #x_test_batch = np.concatenate((xb_mix, X1, xb_mix_2, X2), 0)   
                                    #y_test_batch = np.concatenate((yb_mix, y1, yb_mix_2, y2), 0)
            
                                    test_acc = test_one_batch(x_test_batch, y_test_batch)
                                    acc_mul_batch_0 = acc_mul_batch_0 + test_acc

                                eva_acc_mul_batch_0 = (acc_mul_batch_0*FLAGS.BATCH_SIZE)/test_mul_batch_num
                                if (eva_acc_mul_batch_0 > old_ave_acc):
                                    old_ave_acc = eva_acc_mul_batch_0

                                    best_model_files = os.path.join(best_model_dir, "best_model")
                                    saved_path       = saver.save(sess, best_model_files)
                                    with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                        text_file.write("{0}\n".format(train_acc))
                                        text_file.write("Save best model\n")
                                        text_file.write("New acc: {0}\n".format(eva_acc_mul_batch_0))
                                        #print("Saved best model during testing to {} at batch {}\n".format(saved_path, current_step))
                                        #print("New acc: {}\n".format(eva_acc_mul_batch_0))
                  
                                    if(old_ave_acc > 0.8):
                                        file_valid_acc   = 0
                                        for nFileValid in range(0,valid_file_num):

                                            valid_file_name = valid_file_list[nFileValid]
                                            valid_file_dir  = valid_dir + '/' + valid_file_name
                                            x_valid_batch   = get_test_batch(valid_file_dir)
                                        
                                            # Call training process
                                            valid_end_output = test_process(x_valid_batch)
                                        
                                            # Compute acc
                                            sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                                            valid_res_reg        = np.argmax(sum_valid_end_output)

                                            if(re.search("01_W_", valid_file_name)):
                                                valid_res_exp  = 0
                                            elif(re.search("02_C_", valid_file_name)):
                                                valid_res_exp  = 1
                                            elif(re.search("03_B_", valid_file_name)):
                                                valid_res_exp  = 2
                                            elif(re.search("04_N_", valid_file_name)):
                                                valid_res_exp  = 3

                                            if(valid_res_reg == valid_res_exp):
                                                file_valid_acc = file_valid_acc + 1 
                                           
                                        # For general report
                                        file_valid_acc  = file_valid_acc*100/valid_file_num
                                        with open(os.path.join(stored_dir,"train_acc_log.txt"), "a") as text_file:
                                            text_file.write("Testing Acc: {0}\n".format(file_valid_acc))
                                            #print("Testing Accuracy During Training: {} % \n".format(file_valid_acc))   

                            # 03/ Save model when finish all EPOCHES    
                            if(nEpoch == FLAGS.NUM_EPOCHS):         
                                print("Break at Final Epoch: {}" .format(current_step))
                                is_break = 1
                            elif(old_ave_acc >= 0.95):
                                print("Break at 0.95")
                                is_break = 1
                    #for nBatch in multi-batches
                #for multi-batches
            #for epoch    
                
        if (is_validating == 1):  #if is_training==0  --> testing
            file_valid_acc   = 0
            fuse_matrix      = np.zeros([FLAGS.N_CLASS, FLAGS.N_CLASS])
            valid_metric_reg = np.zeros(FLAGS.N_VALID)
            valid_metric_exp = np.zeros(FLAGS.N_VALID)

            for nFileValid in range(0,valid_file_num):

                valid_file_name = valid_file_list[nFileValid]
                valid_file_dir  = valid_dir + '/' + valid_file_name
                x_valid_batch   = get_test_batch(valid_file_dir)
            
                # Call training process
                valid_end_output = test_process(x_valid_batch)
            
                # Compute acc
                sum_valid_end_output = np.sum(valid_end_output, axis=0) #1xnClass
                valid_res_reg        = np.argmax(sum_valid_end_output)

                if(re.search("01_W_", valid_file_name)):
                    valid_res_exp  = 0
                elif(re.search("02_C_", valid_file_name)):
                    valid_res_exp  = 1
                elif(re.search("03_B_", valid_file_name)):
                    valid_res_exp  = 2
                elif(re.search("04_N_", valid_file_name)):
                    valid_res_exp  = 3


                valid_metric_reg[nFileValid] = int(valid_res_reg)
                valid_metric_exp[nFileValid] = int(valid_res_exp)

                #For general report
                fuse_matrix[valid_res_exp, valid_res_reg] = fuse_matrix[valid_res_exp, valid_res_reg] + 1
                if(valid_res_reg == valid_res_exp):
                    file_valid_acc = file_valid_acc + 1 
               
            # For general report
            file_valid_acc  = file_valid_acc*100/valid_file_num
            print("Testing Accuracy: {} % \n".format(file_valid_acc))   

            #for sklearn metric
            print("Classification report for classifier \n%s\n"
                  % (metrics.classification_report(valid_metric_exp, valid_metric_reg)))
            cm = metrics.confusion_matrix(valid_metric_exp, valid_metric_reg)
            print("Confusion matrix:\n%s" % cm)

            with open(os.path.join(stored_dir,"valid_acc_log.txt"), "a") as text_file:
                text_file.write("========================== VALIDATING ONLY =========================================== \n\n")
                text_file.write("On File Final Accuracy:  {}%\n".format(file_valid_acc))
                text_file.write("{0} \n".format(fuse_matrix))
                text_file.write("========================================================================== \n\n")
tf.contrib.summary
tf.compat.v1.summary.SummaryDescription
tf.summary.all_v2_summary_ops()