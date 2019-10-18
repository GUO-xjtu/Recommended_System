import numpy as np
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


class DigitalClassification(object):
    def __init__(self):
        self.__test_size = 0.20  # 测试集所占比例
        self.__random_state = 15  # 随机数种子
        # 数据加载
        self.digits = load_digits()
        self.data = self.digits.data

    # 展示数据
    def show_image(self):
        print(self.data.shape)  # 数据探索
        print(self.digits.images[0])  # 查看第一幅图像
        print(self.digits.target[0])  # 查看第一幅图片表示的数字含义

        # 显示第一幅图像
        plt.gray()
        plt.imshow(self.digits.images[0])
        plt.show()

    # 数据分割
    def partition_data(self):
        train_x, test_x, train_y, test_y = train_test_split(self.data, self.digits.target, test_size=self.__test_size, random_state=self.__random_state)
        return train_x, test_x, train_y, test_y

    # 使用TPOT进行分类
    def tpot_prediction(self, train_x, test_x, train_y, test_y):
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        tpot.fit(train_x, train_y)
        score = tpot.score(test_x, test_y)
        print('准确率为：%.4f' % score)
        tpot.export("tpot_pipline.py")

    def main(self):
        train_x, test_x, train_y, test_y = self.partition_data()
        self.tpot_prediction(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    DC = DigitalClassification()
    DC.main()