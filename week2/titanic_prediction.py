# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import sklearn.preprocessing as preprocessing


class TitanPrediction():
    def __init__(self):
        """
        运行之前请先修改字体文件路径，在week2/data/msyhbd.ttf中，使用绝对路径
        """
        self.train_data = pd.read_csv(r'/Users/apple/PycharmProjects/homework/week2/data/train.csv')
        self.test_data = pd.read_csv(r'/Users/apple/PycharmProjects/homework/week2/data/test.csv')
        # 定义自定义字体
        self.myfont = FontProperties(fname='/Users/apple/PycharmProjects/homework/week2/data/msyhbd.ttf')

    # 数据探索
    def explore_data(self):
        """
        Passenderld(乘客ID) name(姓名) sex(性别) age(年龄)  Pclass(舱位) Ticket(船票信息) Fare(票价)
        Cabin(客舱) Embarked(登船港口) Parch(父母与小孩个数) sibSp(堂兄弟/妹个数)
        :return: Age缺失较多，Embarked缺失两个，cabin缺失很多（若作为特征加入的话，可能会产生噪音，舍弃）
                Fare中有多个0值
        """
        print('-' * 30 + '数 据 探 索' + '-' * 30)
        pd.set_option('display.max_columns', None)  # 显示所有列
        print('查看数据信息：列名  非空个数  类型等')
        print(self.train_data.info())
        print('-'*30)
        print('查看数据摘要')
        print(self.train_data.describe())
        print('-'*30)
        print('查看离散数据分布')
        print(self.train_data.describe(include=['O']))
        print('-'*30)
        print('查看前10条数据')
        print(self.train_data.head(10))
        print('-'*30)
        print('查看后10条数据')
        print(self.train_data.tail(10))

    # 乘客各属性分布
    def attribute(self):
        """
        有结果看出：只有300多人获救；
                  3等舱的乘客最多；
                  遇难乘客和获救乘客的年龄广度都很大，其中获救的乘客中有80岁的老人。
                  1等舱40岁的乘客最多，2等舱30岁的乘客最多，三等舱25左右的乘客最多。
                  s登船口岸的人数最多，可能和财富值有关。
        :return:
        """
        print('-' * 30 + '乘 客 各 属 性 分 布' + '-' * 30)
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图标颜色alpha参数

        plt.subplot2grid((2, 3), (0, 0))
        self.train_data.Survived.value_counts().plot(kind='bar')  # 柱状图
        plt.title(u"获救情况 （1为获救）", fontproperties=self.myfont)  # 标题
        plt.ylabel(u"人数", fontproperties=self.myfont)

        plt.subplot2grid((2, 3), (0, 1))
        self.train_data.Pclass.value_counts().plot(kind="bar")
        plt.ylabel(u"人数", fontproperties=self.myfont)
        plt.title(u"乘客舱位分布", fontproperties=self.myfont)

        plt.subplot2grid((2, 3), (0, 2))
        plt.scatter(self.train_data.Survived, self.train_data.Age)
        plt.ylabel(u"年龄", fontproperties=self.myfont)
        plt.grid(b=True, which='major', axis='y')
        plt.title(u"按年龄看获救分布 (1为获救)", fontproperties=self.myfont)
        plt.subplot2grid((2, 3), (1, 0), colspan=2)
        self.train_data.Age[self.train_data.Pclass == 1].plot(kind='kde')
        self.train_data.Age[self.train_data.Pclass == 2].plot(kind='kde')
        self.train_data.Age[self.train_data.Pclass == 3].plot(kind='kde')
        plt.xlabel(u"年龄", fontproperties=self.myfont)
        plt.ylabel(u"密度", fontproperties=self.myfont)
        plt.title(u"各舱位的乘客年龄分布", fontproperties=self.myfont)

        plt.legend((u'1', u'2', u'3'), loc='best')
        plt.subplot2grid((2, 3), (1, 2))
        self.train_data.Embarked.value_counts().plot(kind='bar')
        plt.title(u"各登船口岸上船人数", fontproperties=self.myfont)
        plt.ylabel(u"人数", fontproperties=self.myfont)
        plt.show()

    # 将获救结果与属性进行关联统计
    def survived(self):
        print('-'*30 + '幸 存 结 果 与 属 性 的 关 联 分 析' + '-'*30)
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图标颜色alpha参数
        # 乘客舱位与获救结果的关系
        not_survivor = self.train_data.Pclass[self.train_data.Survived == 0].value_counts()
        survivor = self.train_data.Pclass[self.train_data.Survived == 1].value_counts()
        df1 = pd.DataFrame({u'1': survivor, u'0': not_survivor})
        df1.plot(kind='bar', stacked=False)
        plt.xlabel(u'舱房等级', fontproperties=self.myfont)
        plt.ylabel(u'人数', fontproperties=self.myfont)
        plt.title(u'各舱位的获救情况', fontproperties=self.myfont)

        # 各性别的获救情况
        survivor_m = self.train_data.Survived[self.train_data.Sex == 'male'].value_counts()
        survivor_f = self.train_data.Survived[self.train_data.Sex == 'female'].value_counts()
        df2 = pd.DataFrame({u'male': survivor_m, u'female': survivor_f})
        df2.plot(kind='bar', stacked=False)
        plt.title(u'各性别的获救情况', fontproperties=self.myfont)
        plt.xlabel(u'性别', fontproperties=self.myfont)
        plt.ylabel(u'人数', fontproperties=self.myfont)
        plt.show()
        # 各舱位下各性别的获救情况
        plt.title(u'各舱位下各性别的获救情况', fontproperties=self.myfont)
        plt.subplot2grid((2, 2), (0, 0))
        self.train_data.Survived[self.train_data.Sex == 'female'][self.train_data.Pclass != 3].value_counts().plot(kind='bar', label='female highclass', color='#FA2479')
        plt.title(u"高等舱 女性", fontproperties=self.myfont)  # 标题
        plt.ylabel(u"人数", fontproperties=self.myfont)

        plt.subplot2grid((2, 2), (0, 1))
        self.train_data.Survived[self.train_data.Sex == 'female'][self.train_data.Pclass == 3].value_counts().plot(
            kind='bar', label='female lowclass', color='pink')
        plt.title(u"低等舱 女性", fontproperties=self.myfont)  # 标题
        plt.ylabel(u"人数", fontproperties=self.myfont)

        plt.subplot2grid((2, 2), (1, 0))
        self.train_data.Survived[self.train_data.Sex == 'male'][self.train_data.Pclass != 3].value_counts().plot(kind='bar', label='male highclass', color='lightblue')
        plt.title(u"高等舱 男性", fontproperties=self.myfont)  # 标题
        plt.ylabel(u"人数", fontproperties=self.myfont)

        plt.subplot2grid((2, 2), (1, 1))
        self.train_data.Survived[self.train_data.Sex == 'male'][self.train_data.Pclass == 3].value_counts().plot(
            kind='bar', label='male lowclass', color='steelblue')
        plt.title(u"低等舱 男性", fontproperties=self.myfont)  # 标题
        plt.ylabel(u"人数", fontproperties=self.myfont)
        plt.show()

    # 填补空缺数据数据
    def set_data(self):
        """
        使用RandomForest来拟合一下缺失的年龄数据(RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，
                再进行average等等来降低过拟合现象，提高结果的机器学习算法）
        使用Fare均值来填补Fare缺失值
        使用Embarked最多的值填补Embarked缺失值
        :return:
        """
        print('-'*30 + '数 据 集 空 缺 填 补' + '-'*30)
        # 选择预测年龄可能需要的特征（'Age': 结果值; 'Fare':票价越大则财富越多，可能年龄越大; 'Parch':父母与小孩的个数越多可能越是中年;
        #                       'SibSp':兄弟姐妹越多可能越是中年; 'Pclass':舱位越高级可能越是中年人, 'Survived': 幸存者普遍是小孩或者年轻女性）
        age_features = ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']
        train_data = self.train_data[age_features]
        test_data = self.test_data[age_features]

        # 将数据分为已知年龄和未知年龄
        known_age = train_data[train_data.Age.notnull()].as_matrix()
        unknown_train_age = train_data[train_data.Age.isnull()].as_matrix()
        unknown_test_age = test_data[test_data.Age.isnull()].as_matrix()

        train_y = known_age[:, 0]  # 训练需要的年龄结果
        train_x = known_age[:, 1:]  # 训练需要的属性值
        test_train_x = unknown_train_age[:, 1:]  # 预测训练集空值年龄需要的属性值
        test_test_x = unknown_test_age[:, 1:]  # 预测测试集空值年龄需要的属性值

        # 构建模型
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(train_x, train_y)
        # 预测
        test_train_y = rfr.predict(test_train_x)
        test_test_y = rfr.predict(test_test_x)

        # 将预测结果添加到空值处
        self.train_data.loc[(self.train_data.Age.isnull()), 'Age'] = test_train_y
        self.test_data.loc[(self.test_data.Age.isnull()), 'Age'] = test_test_y

        # 使用票价均值填补票价空值
        self.train_data['Fare'].fillna(self.train_data['Fare'].mean(), inplace=True)
        self.test_data['Fare'].fillna(self.test_data['Fare'].mean(), inplace=True)

        # 使用Embarked最多的填补Embarked空值
        self.train_data['Embarked'].fillna(self.train_data['Embarked'].value_counts(), inplace=True)
        self.test_data['Embarked'].fillna(self.train_data['Embarked'].value_counts(), inplace=True)

    # 特征因子化并对数据做一个scaling
    def transform_data(self):
        """
        因子化：将原本只有一个属性维度的类目属性转换为具有多个属性维度的数值属性
               需要转换的类目有：Embarked，Sex，Pclass
        scaling：由于Age和Fare两个属性值在不同乘客之间的差值很大，在做梯度下降时，可能会影响收敛速度甚至不收敛。
                 因此将这两个属性特征化到[-1, 1]之间。
        :return:
        """
        print('-'*30 + '特征因子化 并 对数据做scaling' + '-'*30)
        # 因子化
        dum_train_Emb = pd.get_dummies(self.train_data['Embarked'], prefix='Embarked')
        dum_test_Emb = pd.get_dummies(self.test_data['Embarked'], prefix='Embarked')

        dum_train_Sex = pd.get_dummies(self.train_data['Sex'], prefix='Sex')
        dum_test_Sex = pd.get_dummies(self.test_data['Sex'], prefix='Sex')

        dum_train_Pclass = pd.get_dummies(self.train_data['Pclass'], prefix='Pclass')
        dum_test_Pclass = pd.get_dummies(self.test_data['Pclass'], prefix='Pclass')

        # 将因子化后的特征属性添加到数据集中，并删除原有特征属性
        self.train_data = pd.concat([self.train_data, dum_train_Sex, dum_train_Pclass, dum_train_Emb], axis=1)
        self.train_data.drop(['Sex', 'Pclass', 'Embarked', 'Name'], axis=1, inplace=True)

        self.test_data = pd.concat([self.test_data, dum_test_Sex, dum_test_Pclass, dum_test_Emb], axis=1)
        self.test_data.drop(['Sex', 'Pclass', 'Embarked', 'Name'], axis=1, inplace=True)
        print(self.train_data.head(10))

        # 将Age和Fare特征属性特征化到[-1, 1]之间
        scaler = preprocessing.StandardScaler()
        train_age_scale_param = scaler.fit(self.train_data[['Age']])
        self.train_data['Age_scaled'] = scaler.fit_transform(self.train_data[['Age']], train_age_scale_param)
        train_fare_scale_param = scaler.fit(self.train_data[['Fare']])
        self.train_data['Fare_scaled'] = scaler.fit_transform(self.train_data[['Fare']], train_fare_scale_param)

        test_age_scale_param = scaler.fit(self.test_data[['Age']])
        self.test_data['Age_scaled'] = scaler.fit_transform(self.test_data[['Age']], test_age_scale_param)
        test_fare_scale_param = scaler.fit(self.test_data[['Fare']])
        self.test_data['Fare_scaled'] = scaler.fit_transform(self.test_data[['Fare']], test_fare_scale_param)

        print(self.train_data['Age_scaled'][:5])
        print(self.test_data['Fare_scaled'][:5])

    # 预测测试集乘客幸存状况
    def prediction(self):
        """
        :return:
                LR模型的准确率为80.92% 交叉验证的准确率为80.59% 参数为：penalty='l1', tol=1e-6
                bagging_cart模型的准确率为96.52% 交叉验证的准确率为80.25% 参数为：n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                         bootstrap_features=False, n_jobs=-1
                TOPT模型的准确率为90.24% 选择的最佳参数为81.48%：RandomForestClassifier(input_matrix, bootstrap=True,
                                                    criterion=entropy, max_features=0.8500000000000001, min_samples_leaf=1, min_samples_split=16, n_estimators=100)

        """
        print('-'*30 + '乘 客 幸 存 情 况 预 测' + '-'*30)
        train_data = self.train_data.filter(regex='Survived|Age_scaled|SibSp|Parch|Fare_scaled|Embarked_.*|Sex_.*|Pclass_.*')
        train_data = train_data.as_matrix()
        test_data = self.test_data.filter(regex='Age_scaled|SibSp|Parch|Fare_scaled|Embarked_.*|Sex_.*|Pclass_.*')
        test_data = test_data.as_matrix()

        train_x = train_data[:, 1:]
        train_y = train_data[:, 0]

        # 使用逻辑回归模型
        model_lr = LogisticRegression(penalty='l1', tol=1e-6)
        model_lr.fit(train_x, train_y)
        predictions = model_lr.predict(test_data)
        result = pd.DataFrame({'PassenderId': self.test_data['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
        print(result[:5])
        # 使用训练集得到模型准确率
        predictions = model_lr.predict(train_x)
        print('lr模型准确率为%.4lf' % accuracy_score(train_y, predictions))
        print('lr模型使用交叉验证的准确率为%.4lf\n\n' % np.mean(cross_val_score(model_lr, train_x, train_y, cv=10)))

        # 使用决策树进行模型融合
        model_cart = DecisionTreeClassifier()
        bagging_cart = BaggingClassifier(model_cart, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                         bootstrap_features=False, n_jobs=-1)
        bagging_cart.fit(train_x, train_y)
        # 使用训练集得出模型准确率
        predictions = bagging_cart.predict(train_x)
        print('bagging_cart模型准确率为%.4lf' % accuracy_score(train_y, predictions))
        print('bagging_cart模型使用交叉验证的准确率为%.4lf\n\n' % np.mean(cross_val_score(bagging_cart, train_x, train_y, cv=3)))

        # 使用TPOT模型
        model_tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        model_tpot.fit(train_x, train_y)
        # 使用训练集得出模型准确率
        predictions = model_tpot.predict(train_x)
        print('TPOT模型准确率为%.4lf' % accuracy_score(train_y, predictions))
        print('TPOT模型使用交叉验证的准确率为%.4lf\n\n' % np.mean(cross_val_score(model_tpot, train_x, train_y, cv=3)))


if __name__ == '__main__':
    TP = TitanPrediction()
    TP.explore_data()
    TP.attribute()
    TP.survived()
    TP.set_data()
    TP.transform_data()
    TP.prediction()
