# 使用LeNet模型对Mnist手写数字进行识别
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


class DigitalClassification(object):
    def __init__(self):
        # 数据生成
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
        test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
        self.train_x = train_x / 255
        self.test_x = test_x / 255
        self.train_y = keras.utils.to_categorical(train_y, 10)
        self.test_y = keras.utils.to_categorical(test_y, 10)
        # 创建序贯模型
        self.model = Sequential()

    # 搭建模型
    def set_model(self):
        # 第一层卷积：6个5*5卷积核，relu激活函数
        self.model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        # 第二层池化：最大池化
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 第三层卷积：16个5*5卷积核，relu激活函数
        self.model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        # 第四层池化：最大池化
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # 将参数进行扁平化，在LeNet5中被称为卷积层，实际上是一层一维向量，和全连接层一样
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        # 全连接层，输出节点个数是84个
        self.model.add(Dense(84, activation='relu'))
        # 输出层用softmax激活函数计算分类概率
        self.model.add(Dense(10, activation='softmax'))

    # 设置损失函数和优化器配置
    def lo_op_config(self):
        self.model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # 训练并评估
    def train_predction(self):
        self.model.fit(self.train_x, self.train_y, batch_size=128, epochs=2, verbose=1, validation_data=(self.test_x, self.test_y))

        score = self.model.evaluate(self.test_x, self.test_y)
        print('误差是：%.4lf' % score[0])
        print('精度是%.4f' % score[1])

    # 主函数
    def main(self):
        self.set_model()
        self.lo_op_config()
        self.train_predction()


if __name__ == '__main__':
    DC = DigitalClassification()
    DC.main()
