import tensorflow as tf
import numpy as np
from cabbage.model import CabbageModel


class CabbageTrainer:

    def __init__(self):
        pass

    @staticmethod
    def test_create(data):
        obj = CabbageModel(data)
        obj.create()
        print('모델 생성')
    @staticmethod
    def test(data):

        X = tf.placeholder(tf.float32, shape=[None, 4])  # 변수가 4개
        Y = tf.placeholder(tf.float32, shape=[None, 1])  # 배추가격은 1개
        W = tf.Variable(tf.random_normal([4, 2]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        # 가설을 설정
        hypothesis = tf.matmul(X, W) + b

        model = tf.global_variables_initializer()
        avg_temp = float(input("평균 온도 : "))
        min_temp = float(input("최저 온도 : "))
        max_temp = float(input("최고 온도 : "))
        rain_fall = float(input("강수량 : "))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(model)
            print("불러오기전")
            saver.restore(sess, "./saved/cabbage_model.ckpt")
            print("불러온후")
            data = ((avg_temp, min_temp, max_temp, rain_fall),)
            arr = np.array(data, dtype=np.float32)
            x_data = arr[0:4]
            dict = sess.run(hypothesis, feed_dict={X: x_data})
            print("예측되는 배추가격 {}원".format(dict[0]))
