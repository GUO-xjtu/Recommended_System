from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DigitalClassification(object):
    """
    1、初始化训练数据的权值分布，每个样本开始时赋予相同的权值1/N
    2、训练弱分类器，若某个样本点已被准确分类，那么在构造下一个训练集中，权值就被降低；
       相反，若该样本没有被准确分类，权值就会增加。然后权值更新过的样本集被用于训练下一个
       分类器。
    3、各个弱分类器的训练郭过程束后，加大分类误差率小的弱分类器的权重，降低分类误差率大的弱分类器权重，
       将各个弱分类器组成强分类器。

    结论：随着算法的推进，每一轮迭代都产生一个新的个体分类器被集成。此时集成分类器的分类误差和测试误差都在下降，
         当个体分类器数量达到一定值时，集成分类器的准确率再一定范围内波动。
         这证明：集成学习能很好的抵抗过拟合，训练集和测试集的表现相似。

    """
    def __init__(self):
        self.__test_size = 0.20  # 测试集所占比例
        self.__random_state = 15  # 随机数种子
        # 数据加载
        self.digits = load_digits()
        self.data = self.digits.data

        # 绘图
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    # 数据分割
    def partition_data(self):

        train_x, test_x, train_y, test_y = train_test_split(self.data, self.digits.target,
                                                            test_size=self.__test_size, random_state=self.__random_state)
        return train_x, test_x, train_y, test_y

    def adaboost_prediction(self, train_x, test_x, train_y, test_y):
        score1 = []
        score2 =[]
        index = []
        for i in range(10, 500):
            model = AdaBoostClassifier(n_estimators=i, learning_rate=0.005)
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            score = accuracy_score(y_pred, test_y)
            index.append(i)
            score1 = list(model.staged_score(train_x, train_y))
            score2 = list(model.staged_score(test_x, test_y))
            print('n_estimators = %d, random forest accuracy: %.4lf' % (i, score))
        self.ax.plot(index, score1, label='train score')  # 返回X，y的分阶段分数。
        self.ax.plot(index, score2, label='test score')
        self.ax.set_xlabel('estimators num')
        self.ax.set_ylabel('score')
        self.ax.legend(loc='best')
        self.ax.set_title('AdaBoostClassifier')
        plt.show()

    # 识别主函数
    def main(self):
        train_x, test_x, train_y, test_y = self.partition_data()
        self.adaboost_prediction(train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    DC = DigitalClassification()
    DC.main()