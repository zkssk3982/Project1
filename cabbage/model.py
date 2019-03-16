import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

class CabbageModel:

    def __init__(self, data):
        self.data = data


    def create(self):
        print("모델진입")

        tf.global_variables_initializer()
        xy = np.array(self.data, dtype=np.float32)
        x_data = xy[:, 1: -1]#변인 4개
        y_data = xy[:, [-1]]#배추가격

        X = tf.placeholder(tf.float32, shape=[None, 4]) #변수가 4개
        Y = tf.placeholder(tf.float32, shape=[None, 1]) #배추가격은 1개
        W = tf.Variable(tf.random_normal([4, 1]), name = 'weight')
        b = tf.Variable(tf.random_normal([1]), name = 'bias')
        #가설을 설정
        print("가설을 설정")
        hypothesis = tf.matmul(X, W) + b
        #비용함수 설정
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, "./saved/cabbage_model.ckpt")
        print("학습한 모델을 저장함")
