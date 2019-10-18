import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class EMAnalyze(object):
    def __init__(self):
        self.aliases = pd.read_csv(r'../week4/data/Aliases.csv')  # 别名和对应ID
        self.emails = pd.read_csv(r'../week4/data/Emails.csv')  # 发件人和收件人信息
        self.persons = pd.read_csv(r'../week4/data/Persons.csv')  # 邮件中所有人的名称和ID
        self.aliases_dict = {}
        self.persons_dict = {}
        self.weight_temp = list()

    def explore_data(self):
        print('显示aliases的前5条')
        print(self.aliases.head())
        print('--'*30)
        print('查看aliases数据信息，列名、非空个数，类型')
        print(self.aliases.info())
        print('--'*30)
        print('查看aliases数据摘要')
        print(self.aliases.describe())
        print('--'*30)
        print('查看aliases数据离散分布')
        print(self.aliases.describe(include=['O']))
        print('==='*40 + '\n\n')
        print('显示persons的前5条')
        print(self.persons.head())
        print('--' * 30)
        print('查看persons数据信息')
        print(self.persons.info())
        print('--' * 30)
        print('查看persons的数据摘要')
        print(self.persons.describe())
        print('==='*40 + '\n\n')
        pd.set_option('display.max_columns', None)  # 显示所有列
        print('查看emails的数据离散分布')
        print(self.persons.describe(include=['O']))
        print('显示emails的前5条')
        print(self.emails.head())
        print('--'*30)
        print('查看emails数据信息')
        print(self.emails.info())
        print('--'*30)
        print('查看emails的数据摘要')
        print(self.emails.describe())
        print('--'*30)
        print('查看emails的数据离散分布')
        print(self.emails.describe(include=['O']))
        print('==='*40 + '\n\n')

    # 名称转换
    def tf_name(self, name):

        name = name.lower()
        name = name.replace(',', '')
        name = name.replace(';', '').split('@')[0]
        if name in self.aliases_dict:
            return self.persons_dict[self.aliases_dict[name]]
        else:
            return name

    # 数据清理
    def clean_data(self):
        """
        emails有数据丢失现象：MetadataTo只有7690行，MetadataFrom有7788行。绘制关系网络图需要这两列信息，
                            因此只需要保留这两列全不为空的行。
        人物名称转换：把大写全改为小写；去掉@后的字母
        边权重：使用互相发邮件的次数来表示
        :return:
        """
        # 删除数据不完整的行
        self.emails = self.emails.dropna(subset=['MetadataTo', 'MetadataFrom'], how='any')
        # print(self.emails.info())

        # 规范人物名称
        for index, item in self.aliases.iterrows():
            self.aliases_dict[item['Alias']] = item['PersonId']
        for index, item in self.persons.iterrows():
            self.persons_dict[item['Id']] = item['Name']
        print('正在规范人物名称...')
        self.emails.MetadataTo = self.emails.MetadataTo.apply(self.tf_name)
        self.emails.MetadataFrom = self.emails.MetadataFrom.apply(self.tf_name)
        print('MetadataFrom', self.emails.MetadataFrom)
        print('MetadataTo', self.emails.MetadataTo)
        # 获取边的权重
        weight_temp = dict()
        print('正在计算各边权值...')
        for row in zip(self.emails.MetadataFrom, self.emails.MetadataTo):
            temp = (row[0], row[1])
            if temp not in weight_temp:
                weight_temp.setdefault(temp, 1)
            else:
                weight_temp[temp] += 1

        self.weight_temp = [(key[0], key[1], val) for key, val in weight_temp.items()]
        print('寄件人 收件人 权重: ', self.weight_temp)

    # 绘制网络图
    def draw_img(self):
        # 创建有向图
        graph = nx.DiGraph()
        # 设置有向图路径和权重
        graph.add_weighted_edges_from(self.weight_temp)
        # 计算节点的PageRank值作为节点属性
        pagerank = nx.pagerank(graph)
        nx.set_node_attributes(graph, name='pagerank', values=pagerank)
        # 设置圆环状布局
        pos = nx.circular_layout(graph)

        # 设置中心放射状布局
        # pos = nx.spring_layout(graph)

        # 设置随机分布
        # pos = nx.random_layout(graph)

        # 设置节点大小
        print('正在生成节点大小...')
        nodesize = [n['pagerank']*10000 for _, n in graph.nodes(data=True)]
        print('nodesize', nodesize)
        # 设置网络中的边长度，用权重衡量
        print('正在生成各边长度...')
        edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
        print('edgesize', edgesize)
        # 绘制完整图谱
        print('正在绘制完整图谱...')
        nx.draw_networkx_nodes(graph, pos=pos, node_size=nodesize, alpha=0.4)  # 绘制节点
        nx.draw_networkx_edges(graph, pos=pos, edge_size=edgesize, alpha=0.2)  # 绘制边
        nx.draw_networkx_labels(graph, pos=pos, font_size=10)  # 绘制节点的层
        plt.show()

        # 设置阈值
        pagerank_thrshould = 0.005
        simple_graph = graph.copy()
        for n, page_rank in graph.nodes(data=True):
            if page_rank['pagerank'] < pagerank_thrshould:
                simple_graph.remove_node(n)

        # 绘制大于阈值的图谱
        print('正在绘制精简图谱...')
        nx.draw_networkx_nodes(simple_graph, pos=pos, node_size=nodesize, alpha=0.4)  # 绘制节点
        nx.draw_networkx_edges(simple_graph, pos=pos, edge_size=edgesize, alpha=0.3)  # 绘制边
        nx.draw_networkx_labels(simple_graph, pos=pos, font_size=15)  # 绘制节点的层
        plt.show()
        print('完成！')


if __name__ == '__main__':
    EA = EMAnalyze()
    # EA.explore_data()
    EA.clean_data()
    EA.draw_img()
