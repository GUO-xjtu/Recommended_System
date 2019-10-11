# coding: utf8
import csv
import pandas as pd
from week3.fp_growth import *


class AnalyzeMarBas(object):
    def __init__(self):
        self.file = '/Users/apple/PycharmProjects/homework/week3/data/Market_Basket_Optimisation.csv'
        self.transactions = []
        self.transactions_hot_encoded = None
        self.frequent_items1 = None
        self.frequent_items2 = None
        self.frequent_items3 = None
        self.rules1 = None
        self.rules2 = None
        self.rules3 = None

    # 修改文件格式
    def set_data(self):
        """
        为每一行transaction增加ID列
        将transaction作为一列
        去除只有一个item的transaction
        :return:
        """
        row = 0
        with open(self.file, 'r') as csv_input:
            with open('/Users/apple/PycharmProjects/homework/week3/data/MarBasOpt.csv', 'w') as csv_output:
                csv_w = csv.writer(csv_output)
                csv_w.writerow(['id', 'transaction'])
                for line in csv.reader(csv_input):
                    # 去除只有一件物品的transaction
                    if len(line) == 1:
                        continue
                    lines = [row]
                    transaction = ''
                    for item in line:
                        transaction += (item + '|')
                    transaction = transaction[:-1]
                    lines.append(transaction)
                    row += 1
                    csv_w.writerow(lines)

    # 创建transaction
    def set_transaction(self):
        pd.set_option('display.max_columns', None)
        transactions = pd.read_csv('/Users/apple/PycharmProjects/homework/week3/data/MarBasOpt.csv')

        # 生成热编码，为使用analyze_rule1准备
        self.transactions_hot_encoded = transactions.drop('transaction', 1).join(
            transactions.transaction.str.get_dummies('|'))
        self.transactions_hot_encoded.set_index(['id'], inplace=True)

        # 生成transaction列表，为analyze_rule2和analyze_rule3准备
        item_series = transactions.set_index('id')['transaction']
        for index, item in item_series.items():
            item = item.split('|')
            self.transactions.append(item)
        print(self.transactions)

    # 使用Aprior挖掘关联规则
    def analyze_rule1(self):
        from mlxtend.frequent_patterns import apriori
        from mlxtend.frequent_patterns import association_rules
        frequent_items = apriori(self.transactions_hot_encoded, min_support=0.02, use_colnames=True)
        frequent_items1 = frequent_items.sort_values(by='support', ascending=False)
        rules1 = association_rules(frequent_items, metric='lift', min_threshold=1)
        rules1 = rules1.sort_values(by='lift', ascending=False)

        self.frequent_items1 = {}
        length_items = len(frequent_items1['support'])
        for i in range(length_items):
            value = frequent_items1['support'][i]
            key = frequent_items1['itemsets'][i]
            self.frequent_items1[key] = value
        self.frequent_items1 = [v for v in sorted(self.frequent_items1.items(), key=lambda b:b[1], reverse=True)]

        self.rules1 = {}
        length_rules = len(rules1['support'])
        for i in range(length_rules):
            key = (rules1['antecedents'][i], rules1['consequents'][i])
            value = rules1['confidence'][i]
            self.rules1[key] = value
        self.rules1 = [v for v in sorted(self.rules1.items(), key=lambda b:b[1], reverse=True)]

        print('挖掘出的频繁项集：\n', self.frequent_items1)
        print('得到的关联规则：\n', self.rules1)

    def analyze_rule2(self):
        from efficient_apriori import apriori
        fre_items, rules2 = apriori(self.transactions, min_support=0.01, min_confidence=0.5)
        self.frequent_items2 = {}
        for key, item in enumerate(fre_items.items()):
            item = [i for i in sorted(item[1].items(), key=lambda a:a[1], reverse=True)]
            self.frequent_items2[key+1] = item
        self.rules2 = rules2
        print('挖掘出的频繁项集：', self.frequent_items2)
        print('得到的关联规则：', self.rules2)

    # 使用FP_Growth挖掘关联规则
    def analyze_rule3(self):
        self.frequent_items3, self.rules3 = fpGrowth(dataSet=self.transactions, minSup=20, minConf=0.5)
        print('挖掘出的频繁项集：', self.frequent_items3)
        print('得到的关联规则：', self.rules3)


if __name__ == '__main__':
    AMB = AnalyzeMarBas()
    AMB.set_data()
    AMB.set_transaction()
    AMB.analyze_rule1()
    print('-'*50)
    AMB.analyze_rule2()
    print('-' * 50)
    AMB.analyze_rule3()
