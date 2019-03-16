from gtsrb.model import Gtsrb
import matplotlib.pyplot as plt

"""
        gt = Gtsrb()
        dataset = gt.read_dataset(
            'images/',
            gt.N_CLASSES,
            gt.RESIZE_IMAGE)
"""
class Trainer:
    def __init__(self, dataset):
        gt = Gtsrb()
        self.dataset = dataset

    def excute(self):
        gt = Gtsrb()
        dataset = self.dataset
        print("X 데이터셋".format(dataset.X.shape))
        print("Y 데이터셋".format(dataset.Y.shape))

        plt.imshow(
            dataset.X[0, :, :, :].reshape(gt.RESIZE_IMAGE)
        )#표본 이미지
        print("레이블: {}".format(dataset.y[0, :]))

        plt.imshow(dataset.X[-1, :, :, :].reshape(gt.RESIZE_IMAGE))#답? 문제? 답? 문제
        print("레이블: {}".format(dataset.y[-1, :]))

        plt.show()