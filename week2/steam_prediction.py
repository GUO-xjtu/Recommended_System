import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SteamPrediction(object):
    def __init__(self):
        data = pd.read_csv('../week2/data/steam-200k.csv', header=None, names=['UserID', 'Game', 'Action',
                                                                            'Hours', 'Not Needed'])
        self.df = data

    def explore_data(self):
        pd.set_option('display.max_columns', None)  # 显示所有列
        print('显示前5条数据')
        print(self.df.head())
        print('=='*20)
        print('显示后5条数据')
        print(self.df.tail())
        print('==' * 20)
        print('显示数据大小')
        print(self.df.shape)
        print('==' * 20)
        print('查看数据信息：列名、非空个数、类型等')
        print(self.df.info())
        print('==' * 20)
        print('查看数据摘要')
        print(self.df.describe())
        print('==' * 20)
        print('查看离散数据分布')
        print(self.df.describe(include=['O']))
        print('=='*20)

    # 数据清洗
    def clean_data(self):
        """
        将Action和Hours合并为Play_Hours
        删除重复项
        删除多余的列
        对数据进行排序
        drop_duplicates(subset=None, keep='first', inplace=False)
            subset 根据指定的列名进行去重，默认整个数据集；
            keep 可选「first last False」即默认保留第一次出现的重复值，并删去其他重复的数据，False是指删去所有重复数据
            inplace 是否对数据集本身进行修改，默认False
        loc([index1, index2], index3) : 提取index为[index1, index2],列名为index3的数据
        astype(): 强制类型转换
        :return:
        """
        pd.set_option('display.max_columns', None)  # 显示所有列

        self.df['Play_Hours'] = self.df['Hours'].astype('float64')
        # 若Action字段值为purchase,则判断Hours是否为1.0，若是，则设置Play_Hours=0
        self.df.loc[(self.df['Action'] == 'purchase') & (self.df['Hours'] == '1.0'), 'Play_Hours'] = 0

        # 删除重复项
        self.df = self.df.drop_duplicates(['UserID', 'Game'], keep='last')

        # 删除多余的列
        self.df = self.df.drop(['Action', 'Hours', 'Not Needed'], axis=1)
        # 对数据从小到大进行排序，df下标也会改变
        self.df.UserID = self.df.UserID.astype('int')
        self.df = self.df.sort_values(['UserID', 'Game', 'Play_Hours'], ascending=True)

        # 数据集探索
        # self.explore_data()

    # 数据可视化
    def plot_attribute(self):
        fig = plt.figure()
        fig.set(alpha=0.2)
        ugc = {}  # 每个用户玩过的游戏数
        ugcs = self.df.groupby('UserID')  # 每个用户玩过的游戏数
        # df1 = pd.DataFrame({u'UserID': ugc})
        # df1.plot(kind='bar', stacked=False)
        # plt.xlabel('UserID')
        # plt.ylabel('game_num')
        # plt.title('User_Game_Num')
        # print(ugc['Game'].count())
        for i in ugcs:
            ugc[i[0]] = len(i[1])
        ugc = [i for i in sorted(ugc.items(), key=lambda d:d[1], reverse=True)][:20]
        print(ugc)
        print('=='*30)
        guc = self.df.groupby('Game')['UserID'].count()  # 每个游戏拥有的用户数
        print(guc)

    def main(self):
        # self.explore_data()
        self.clean_data()
        self.plot_attribute()


if __name__ == '__main__':
    SP = SteamPrediction()
    SP.main()