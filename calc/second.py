import tensorflow as tf


class Second:

    def __init__(self):
        pass

    @staticmethod
    def excute():
        sess = tf.Session()
        saver = tf.train.import_meta_graph('saved/model-1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./saved/'))
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1: 0")
        w2 = graph.get_tensor_by_name("w2: 0")
        feed_dict = {w1: 13.0, w2: 17.0}
        op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

        print(sess.run(op_to_restore, feed_dict))

