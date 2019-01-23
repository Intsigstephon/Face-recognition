#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:04:27 2018

@author: houliang
"""

import numpy as np
import tensorflow as tf
import input_getbatch
import model_testc
import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

N_CLASSES = 2
IMG_W = 96
IMG_H = 96

CAPACITY = 1000
MAX_STEP = 15
BATCH_SIZE = 200
logs_train_dir = '../Train_Network/checkpoint/model_jiayin'
             
def run_testing(imgdir):
    
    val_dir = imgdir 
    print('Test Images:')
    val, val_label = input_getbatch.get_files(val_dir)
    val_batch, val_label_batch = input_getbatch.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
    
    x  = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 96, 96, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = model_testc.inference(x, BATCH_SIZE, N_CLASSES, 1.0)
    loss = model_testc.losses(logits, y_) 
    acc = model_testc.evaluation(logits, y_)
    total_loss = 0
    total_acc = 0
    
    saver = tf.train.Saver()      
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                   break

                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict = {x:val_images, y_:val_labels})

                total_loss = total_loss + val_loss
                total_acc = total_acc + val_acc

                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc * 100.0))
                    
                if step + 1 == MAX_STEP:
                    print('Total accuracy = %.5f%%'%((total_acc / (step+1) * 100.0)))
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)

def getMinIndex(fpr, target):
    """
    given fpr, get index 
    """
    min_dis = 1
    min_index = 0

    for i in range(len(fpr[1])):
        val = fpr[1][i]
        tmp_dis = abs(val - target)
        if tmp_dis < min_dis:
            min_dis = tmp_dis
            min_index = i

    return min_index

def ROC(y_rslts, y_labels):
    """
    given y_rslts
    """
    y_score = np.asarray(y_rslts)
    y_test  = np.asarray(y_labels)

    #print(y_rslts)
    #print(y_labels)

    #Compute ROC curve and ROC area for each class
    n_classes = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ths = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], ths[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    index_0 = getMinIndex(fpr, 0.2)
    print("0.2 fpr: {},    threshold: {}".format(tpr[1][index_0], ths[1][index_0]))

    index_1 = getMinIndex(fpr, 0.1)
    print("0.1 fpr: {},    threshold: {}".format(tpr[1][index_1], ths[1][index_1]))

    index_2 = getMinIndex(fpr, 0.01)
    print("0.01 fpr: {},   threshold: {}".format(tpr[1][index_2], ths[1][index_2]))

    #set thresh = 0.5;
    index_3 = getMinIndex(ths, 0.5)  
    print("fpr:{} , tpr: {},  threshold: {}".format(fpr[1][index_3], tpr[1][index_3], ths[1][index_3]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot of a ROC curve for a specific class＼
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',lw=lw, label='ROC real image curve (area = %0.2f)' % roc_auc[1])
    #plt.plot(fpr[0], tpr[0], color='blue',lw=lw, label='ROC fake image curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def run_testing2(test_file):
    pos_gt_num = 0
    neg_gt_num = 0
    pos_det_num = 0
    neg_det_num = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    #for ROC
    y_rslts = []
    y_labels = []

    print('Test Images:')
    val, val_label = input_getbatch.get_files_bytxt(test_file, os.path.dirname(test_file), ' ')    #image, label
    print("test data num is: {}".format(len(val)))

    val_batch, val_label_batch = input_getbatch.get_batch(val,
                                                  val_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)

    x  = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 96, 96, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    
    logits = model_testc.inference(x, BATCH_SIZE, N_CLASSES, 1.0)
    loss = model_testc.losses(logits, y_) 
    acc = model_testc.evaluation(logits, y_)

    #begin to summary
    total_loss = 0
    total_acc = 0
    
    saver = tf.train.Saver()      
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                   break

                val_images, val_labels = sess.run([val_batch, val_label_batch])
                val_loss, val_acc, val_logits = sess.run([loss, acc, logits], feed_dict = {x:val_images, y_:val_labels})
                val_rslts = np.argmax(val_logits, axis=1)

                #summary                
                for i in range(len(val_labels)):
                    label = val_labels[i]
                    rslt  = val_rslts[i]
                    if rslt == 1:   
                        if label:
                            TP +=1   #lable=1, rslt=1
                        else:
                            FP +=1   #label=0, rslt=1
                    else:
                        if label:
                            FN +=1   #label=1, rslt=0
                        else:
                            TN +=1   #label=0, rslt=0
                    
                    #rslt
                    __logit = val_logits[i]
                    __logit = np.exp(__logit)
                    __logit = __logit / sum(__logit)
                    y_rslts.append([__logit[0], __logit[1]])       #first: neg posibility, second: pos posibility

                    #label
                    tmp = [0, 0]
                    tmp[label] = 1              #1. [0,1]; 0. [1, 0]
                    y_labels.append(tmp)
                
                #print(y_rslts)
                #print(y_labels)
                    
                neg_gt_num =  FP + TN
                pos_gt_num =  TP + FN
                neg_det_num = TN + FN
                pos_det_num = TP + FP  

                #get total loss and total acc
                total_loss = total_loss + val_loss
                total_acc = total_acc + val_acc

                if step % 1 == 0 or (step + 1) == MAX_STEP:
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc * 100.0))
                    
                if step + 1 == MAX_STEP:
                    print('Total accuracy = %.5f%%' % ((total_acc / (step + 1) * 100.0)))
                    print('Total negative truth num: {}'.format(neg_gt_num))
                    print('Total positvie truth num: {}'.format(pos_gt_num))
                    print('Total negative detect num: {}'.format(neg_det_num))
                    print('Total positvie detect num: {}'.format(pos_det_num))

                    #FP, TN, FN, TP
                    print("FP: {}".format(FP))
                    print("TN: {}".format(TN))
                    print("FN: {}".format(FN))
                    print("TP: {}".format(TP))

                    #Total acc; Pos acc, Neg acc
                    print("Total acc: {}".format((TN + TP) / (neg_gt_num + pos_gt_num + 0.0)))  #90.83%
                    print("Neg acc", 1.0 * TN / neg_gt_num)   #neg recall
                    print("Pos acc", 1.0 * TP / pos_gt_num)   #pos recall

                    #Draw ROC
                    ROC(y_rslts, y_labels)
                       
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)

if __name__=="__main__":

    # if len(sys.argv) < 2:
    #     imgdir = '/media/stephon/_H/Face_Anti_poofing/Datasets/NUAA/raw' 
    #     print("NO imgdir specified!!!\nDefault: {}".format(imgdir))
    # else:
    #     imgdir = sys.argv[1]
    #     print(imgdir)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    #run_testing(imgdir)
    test_file = '/media/stephon/_H/Face_Anti_poofing/Projects/anti-spoofing/dataset/align_images/test/test.txt'       
    run_testing2(test_file)  

    #结果记录: 60000: 
    #  0.2 fpr: 0.970985492746,    threshold: 0.910939991474
    #  0.1 fpr: 0.946973486743,    threshold: 0.999968886375

    #结果记录: 62000: 
    #  0.2 fpr: 0.973973973974,    threshold: 0.174229606986
    #  0.1 fpr: 0.950950950951,    threshold: 0.999514877796

    #目前的最好模型: 迭代了62000次; 达到fpr: 0.1, tpr: 95.10%; fpr:0.2, tpr: 97.40%;
