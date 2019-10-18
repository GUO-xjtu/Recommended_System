import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
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

    # LinearSVC分类器
    def linearsvc_predict(self, train_x, test_x, train_y, test_y):
        """
        仅支持线性分类，默认l2正则化，
        dual:选择算法以解决双优化或原始优化问题。 当n_samples> n_features时，首选dual = False。
        """
        model = svm.LinearSVC(dual=False)
        model.fit(train_x, train_y)
        z = model.predict(test_x)
        print('准确率：', np.sum(z == test_y)/z.size)

    # SVC分类器
    def svc_predict(self, train_x, test_x, train_y, test_y):
        """
        gamma:支持向量机的间隔，即是超平面距离不同类别的最小距离
              选择auto时 --> gamma = 1/feature_num
              选择scale时 --> gamma = 1/(feature_num * x.std())，即特征数目乘以样本标准差分之一
        C:松弛变量的系数，称为惩罚系数，当C越大，说明该模型对分类错误更加容忍，也就是为了避免过拟合
          范围是1到正无穷
        decision_function_shape:
              ovr: 将一个类别与其他所有类别进行划分
              ovo: 类别两两划分
        kernel: 核函数的选择
              当样本线性可分时，可选linear线性核函数
              当线性不可分时，可选rbf(高斯核函数)，poly(多项式核函数)，sigmoid核函数
        """
        model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='linear')
        model.fit(train_x, train_y)
        z = model.predict(test_x)
        print('准确率：', np.sum(z == test_y) / z.size)

    # NuSVC分类器
    def nusvc_predict(self, train_x, test_x, train_y, test_y):
        """
        SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同
        C: 范围是0到1
        nu的一个好特性是它与支持向量的比率和训练误差的比率有关。

        """
        model = svm.NuSVC(gamma='scale', decision_function_shape='ovr', kernel='rbf')
        model.fit(train_x, train_y)
        z = model.predict(test_x)
        print('准确率：', np.sum(z == test_y) / z.size)

    def main(self):
        # self.show_image()
        train_x, test_x, train_y, test_y = self.partition_data()
        self.linearsvc_predict(train_x, test_x, train_y, test_y)
        self.svc_predict(train_x, test_x, train_y, test_y)
        self.nusvc_predict(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    DC = DigitalClassification()
    DC.main()