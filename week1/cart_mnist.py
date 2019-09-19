#  使用CART进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
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

    def partition_data(self):
        """
        数据分割
        train_test_split (*arrays，test_size, train_size, rondom_state=None, shuffle=True, stratify=None)

        arrays：特征数据和标签数据（array，list，dataframe等类型），要求所有数据长度相同。
        test_size / train_size: 测试集/训练集的大小，若输入小数表示比例，若输入整数表示数据个数。
        random_state: 随机数种子，即该组随机数的编号，在需要重复实验的时候，保证得到一组一样的随机数
        shuffle：是否打乱数据的顺序，再划分，默认True。
        stratify：none或者array/series类型的数据，表示按这列进行分层采样。
        :return: 返回分类好的训练集（train_x,train_y）和测试集（test_x, test_y）
        """
        train_x, test_x, train_y, test_y = train_test_split(self.data, self.digits.target,
                                                            test_size=self.__test_size, random_state=self.__random_state)
        return train_x, test_x, train_y, test_y

    def normalization_data(self, train_x, test_x):
        """
        数据规范化
        :param train_x:
        :param test_x:
        :return: 返回规范后的训练集（train_ss_x）和测试集（test_ss_x）
        """
        # ss = preprocessing.MaxAbsScaler()  # 准确率 86.67 数据会被规模化到[-1,1]之间。也就是特征中，所有数据都会除以最大值。这个方法对那些已经中心化均值维0或者稀疏的数据有意义。
        # ss = preprocessing.StandardScaler()  # 准确率 86.97 可以传入两个参数：with_mean,with_std。这两个都是布尔类型的参数，默认下都是true，也能自定义false,即不要均值中心化或不要方差规模化为1
        # ss = preprocessing.Normalizer()  # 准确率：85.83 正则化是将样本在向量模型上的一个转换，经常被使用在分类与聚类中。
        ss = preprocessing.Binarizer(threshold=4.0)  # 准确率88.33 特征的二值化是指将数值型的特征数据转换成布尔类型的值。可以使用实用类Binarizer, 默认是根据0来二值化，大于0的都标记为1，小于等于0的都标记为0。
        train_ss_x = ss.fit_transform(train_x)
        test_ss_x = ss.transform(test_x)  # 用同样的参数标准化测试集，使得测试集和训练集之间有可比性
        return train_ss_x, test_ss_x

    # 创建CART，并训练
    def cart_prediction(self, train_x, train_y, test_x, test_y):
        cart = DecisionTreeClassifier()
        cart.fit(train_x, train_y)
        predict_y = cart.predict(test_x)
        print('CART准确率为：%0.4lf' % accuracy_score(predict_y, test_y))

    # 主函数
    def main(self):
        self.show_image()
        train_x, test_x, train_y, test_y = self.partition_data()
        train_x, test_x = self.normalization_data(train_x, test_x)
        self.cart_prediction(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    # 测试出的准确率最高为88.33(使用二值化(threshold=4.0)进行预处理，test_size=0.20,random_state=15)
    DC = DigitalClassification()
    DC.main()
