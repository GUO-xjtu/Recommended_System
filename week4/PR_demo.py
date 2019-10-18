import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class PrDemo(object):
    def __init__(self):
        self.edges = [('a', 'b'), ('a', 'f'), ('a', 'e'), ('a', 'd'), ('b', 'c'), ('c', 'e'), ('d', 'e'), ('d', 'a'),
                      ('d', 'c'), ('e', 'b'), ('e', 'c'), ('f', 'd')]
        self.edges = sorted(self.edges)
        self.node = list()  # 节点列表
        self.array = None  # 转移矩阵
        self.weight = None  # 权重矩阵
        print('图为：', self.edges)

    # 获取节点
    def get_node(self):
        node = set()
        for i in self.edges:
            node.add(i[0])
            node.add(i[1])
        self.node = sorted(node)
        self.weight = np.array(([1/len(self.node)]*len(self.node)))
        print('节点为：', self.node)
        print('边的权重为：', self.weight)

    # 获取边
    def get_edge(self):
        j = 1
        edge = []
        edges = []
        le = len(self.edges)
        for i in range(le):
            if j == le:
                edge.append(self.edges[i])
                edges.append(edge)
                break
            if self.edges[i][0] == self.edges[j][0]:
                edge.append(self.edges[i])
            else:
                edge.append(self.edges[i])
                edges.append(edge)
                edge = []
            j += 1
        self.edges = edges
        print('节点出度：', self.edges)

    # 计算转移矩阵
    def set_array(self):
        array = []
        arr = []
        for edge in self.edges:
            weight = 1/len(edge)
            k = 0
            for node in self.node:
                if edge[k][1] == node:
                    arr.append(weight)
                    if k < len(edge)-1:
                        k += 1
                else:
                    arr.append(0)

            array.append(arr)
            arr = []
        self.array = np.array(array).T
        print('转移矩阵为：', self.array)

    # 简单模型
    def simple_model(self):
        for i in range(100):
            w = np.dot(self.array, self.weight)
            self.weight = w
        print('简单模型pr：', self.weight)

    # 随机浏览模型
    def random_work(self):
        p = 0.85
        for i in range(100):
            w = (1-p)/len(self.node) + p*(np.dot(self.array, self.weight))
            self.weight = w
        print('随机浏览模型pr：', self.weight)

    # PageRank工具使用
    def pr_work(self):
        G = nx.DiGraph()  # 创建有向图
        for edge in self.edges:
            G.add_edges_from(edge)  # 添加边集合
        # 有向图可视化
        layout = nx.spring_layout(G)
        nx.draw(G, layout, with_labels=True, hold=False)
        plt.show()

        # 计算简化模型pr值
        simple_pr = nx.pagerank(G, alpha=1)
        print('简化模型pr:', simple_pr)
        # 计算随机模型pr值
        random_pr = nx.pagerank(G, alpha=0.85)
        print('随机模型pr:', random_pr)


if __name__ == '__main__':
    PRD = PrDemo()
    PRD.get_node()
    PRD.get_edge()
    PRD.set_array()
    PRD.simple_model()
    PRD.random_work()
    PRD.pr_work()

