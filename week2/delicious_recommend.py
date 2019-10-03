import random
import operator
import math


class DeliciousRecommend():
    """
    覆盖率：推荐的item数量 / 训练集中item数量
    多样性：推荐结果中各个item之间的余弦相似度，再求平均
    平均热门程度：log(1+item的用户数) / 参与测试的用户数
    """
    def __init__(self):
        self.filename = "./data/user_taggedbookmarks-timestamps.dat"
        self.ratio = 0.2  # 测试集所占的比例大致为0.2
        self.seed = 100  # 随机数种子
        self.records = {}  # 保存用户对书籍打的标签
        self.user_tags = {}  # 用户打了哪些标签
        self.tag_items = {}  # 标签包含哪些书籍
        self.user_items = {}  # 用户给哪些书籍打了标签
        self.tag_users = {}  # 标签被哪些用户使用过
        self.item_tags = {}  # 书籍被打上了哪些标签
        self.item_users = {}  # 书籍被哪些用户打了标签
        self.train = {}
        self.test = {}


    # 加载数据
    def loadData(self):
        print("开始数据加载...")
        fi = open(self.filename)
        lineNum = 0
        for line in fi:
            lineNum += 1
            if lineNum == 1:
                continue
            uid, iid, tag, timestamp = line.split('\t')

            # 让id在数组中对应下标从0开始
            uid = int(uid) - 1
            iid = int(iid) - 1
            tag = int(tag) - 1

            # 存储某用户给某本书打的某些标签
            self.records.setdefault(uid, {})
            self.records[uid].setdefault(iid, [])
            self.records[uid][iid].append(tag)
        fi.close()

        print("数据集大小为 %d " % (lineNum))
        print("打过标签(tag)的人数：%d" % (len(self.records)))
        print("数据加载完成！")

    # 将数据集拆分为训练集和测试集
    def randomlySplitData(self):
        random.seed(self.seed)  # 当设置了seed，则若seed相同，那么生成的随机数就相同
        print(random.random())
        random.seed(self.seed)
        print(random.random())  # 产生[0,1)之间的随机浮点数

        for user in self.records.keys():
            for item in self.records[user].keys():
                # 若产生的随机数小于0.2，那么归为测试集
                if random.random() < self.ratio:
                    self.test.setdefault(user, {})
                    self.test[user].setdefault(item, [])
                    for tag in self.records[user][item]:
                        self.test[user][item].append(tag)
                else:
                    self.train.setdefault(user, {})
                    self.train[user].setdefault(item, [])
                    for tag in self.records[user][item]:
                        self.train[user][item].append(tag)
        print("训练集样本数 %d, 测试集样本数 %d" % (len(self.train), len(self.test)))

    # 设置矩阵mat[index, item] = 1
    def _addValueToMat(self, mat, index, item, value=1):
        if index not in mat:
            mat.setdefault(index, {})
            mat[index].setdefault(item, value)
        else:
            if item not in mat[index]:
                mat[index][item] = value
            else:
                mat[index][item] += value

    # 使用训练集，初始化user_tags, tag_items, user_items
    def initStat(self):

        for user, items in self.train.items():  # 以列表返回可遍历的(键、值)元组
            # print(user, items)  # 用户user给哪些书籍打了哪些标签
            for item, tags in items.items():
                # print(item, tags)  # 书籍item都被打上了哪些标签
                for tag in tags:
                    # 用户和标签的关系
                    self._addValueToMat(self.user_tags, user, tag, 1)
                    # 标签和书籍的关系
                    self._addValueToMat(self.tag_items, tag, item, 1)
                    # 用户和书籍的关系
                    self._addValueToMat(self.user_items, user, item, 1)
                    # 标签和用户的关系
                    self._addValueToMat(self.tag_users, tag, user, 1)
                    # 书籍和标签的关系
                    self._addValueToMat(self.item_tags, item, tag, 1)
                    # 书籍和用户的关系
                    self._addValueToMat(self.item_users, item, user, 1)

        print("user_tags, tad_items, user_items初始化完成！")
        print("user_tags大小为 %d, tag_items大小为 %d, user_items大小为 %d" %(len(self.user_tags), len(self.tag_items), len(self.user_items)))

    # 对用户user推荐Top-N
    def recommend(self, user, N):
        stb_recommend_items = dict()  # SimpleTagBased算法推荐清单
        ntb_recommend_items = dict()  # NormTagBased算法推荐清单
        tfidf_recommend_items = dict()  # TagBased-TFIDF算法推荐清单
        tag_items = self.user_items[user]
        lut = len(self.user_tags[user])
        # print('该用户使用了%d种tag' % lut)
        for tag, wut in self.user_tags[user].items():
            # print('该用户将这个tag总共使用%d次' % wut)
            lti = len(self.tag_items[tag])
            # print('tag标签被打在了%d种item上' % lti)
            ltu = len(self.tag_users[tag])
            # print('tag标签被%d个用户使用过' % ltu)
            for item, wit in self.tag_items[tag].items():
                if item in tag_items:
                    continue
                # print('item上打了tag标签有%d次' % wit)
                # SimpleTagBased
                if item not in stb_recommend_items:
                    stb_recommend_items[item] = wut * wit
                elif item in stb_recommend_items:
                    stb_recommend_items[item] += wut * wit
                # NormTagBased
                if item not in ntb_recommend_items:
                    ntb_recommend_items[item] = ((wut / lut) * (wit / lti))
                elif item in ntb_recommend_items:
                    ntb_recommend_items[item] += ((wut / lut) * (wit / lti))
                # TagBased-TFIDF
                if item not in tfidf_recommend_items:
                    tfidf_recommend_items[item] = wut * wit / math.log(1 + ltu)
                elif item in tfidf_recommend_items:
                    tfidf_recommend_items[item] += wut * wit / math.log(1 + ltu)
        return sorted(stb_recommend_items.items(), key=operator.itemgetter(1), reverse=True)[:N], \
               sorted(ntb_recommend_items.items(), key=operator.itemgetter(1), reverse=True)[:N], \
               sorted(tfidf_recommend_items.items(), key=operator.itemgetter(1), reverse=True)[:N]

    # 物品i和j的余弦相似度
    def cosineSim(self, i, j):
        ret = 0
        ni, nj = 0, 0
        for tag, wtt in self.item_tags[i].items():
            ni += wtt * wtt
            if tag in self.item_tags[j]:
                ret += wtt * self.item_tags[j][tag]

        for tag, wtt in self.item_tags[j].items():
            nj += wtt * wtt

        if ret == 0:
            return 0
        return ret/math.sqrt(ni * nj)

    # 预测准确率\召回率\覆盖率\多样性\平均热门程度
    def prediction(self, N):
        stb, stb_ret, stb_ap = 0, 0, 0
        ntb, ntb_ret, ntb_ap = 0, 0, 0
        tfidf, tfidf_ret, tfidf_ap = 0, 0, 0
        total_items = 0  # 训练集中某用户的item总数
        test_items = 0  # 测试集中使用到的用户数
        precision = 0
        recall = 0
        for user, items in self.test.items():
            if user not in self.train:
                continue
            test_items += 1
            total_items += len(self.train[user])

            stb_rank, ntb_rank, tfidf_rank = self.recommend(user, N)

            for item_i, score in stb_rank:
                if item_i in items:
                    stb += 1
                    stb_ap += math.log(1 + len(self.item_users[item_i]))
                for item_j, score in stb_rank:
                    if item_j != item_i:
                        stb_ret += self.cosineSim(item_i, item_j)
            for item_i, score in ntb_rank:
                if item_i in items:
                    ntb += 1
                    ntb_ap += math.log(1 + len(self.item_users[item_i]))
                for item_j, score in ntb_rank:
                    if item_j != item_i:
                        ntb_ret += self.cosineSim(item_i, item_j)

            for item_i, score in tfidf_rank:
                if item_i in items:
                    tfidf += 1
                    tfidf_ap += math.log(1 + len(self.item_users[item_i]))
                for item_j,score in tfidf_rank:
                    if item_j != item_i:
                        tfidf_ret += self.cosineSim(item_i, item_j)

            recall += len(items)
            precision += N
        stb_diversity, stb_aver = (stb_ret / ((N*N-N)*test_items*1.0)) * 100, (stb_ap / (test_items * N * 1.0)) * 100
        ntb_diversity, ntb_aver = (ntb_ret / ((N*N-N)*test_items*1.0)) * 100, (ntb_ap / (test_items * N * 1.0)) * 100
        tfidf_diversity, tfidf_aver = (tfidf_ret / ((N*N-N)*test_items*1.0)) * 100, (tfidf_ap / (test_items * N * 1.0)) * 100
        coverage = (precision/(total_items*1.0))*100
        re_stb, pre_stb = (stb/(recall*1.0))*100, (stb/(precision*1.0))*100
        re_ntb, pre_ntb = (ntb/(recall*1.0))*100, (ntb/(precision*1.0))*100
        re_tfidf, pre_tfidf = (tfidf/(recall*1.0))*100, (tfidf/(precision*1.0))*100
        return stb, re_stb, pre_stb, stb_diversity, stb_aver, ntb, re_ntb, pre_ntb, ntb_diversity, ntb_aver, \
               tfidf, re_tfidf, pre_tfidf, tfidf_diversity, tfidf_aver, coverage

    # 使用测试集对推荐结果进行预测
    def testRecommend(self):
        print('\n' + '-'*40+'--推 荐 结 果 评 估--'+'-'*40)
        print('%3s %20s %9s %8s %9s %9s %9s %8s' % ('算法', 'N', '命中数', '召回率', '精确率', '覆盖率', '多样性', '平均热门程度'))
        for n in [5, 10, 20, 40, 60, 80, 100]:
            stb, re_stb, pre_stb, stb_diversity, stb_aver, ntb, re_ntb, pre_ntb, ntb_diversity, ntb_aver, \
            tfidf, re_tfidf, pre_tfidf, tfidf_diversity, tfidf_aver, coverage = self.prediction(n)

            print('%3s %10d %10s %10.3f%% %10.3f%% %10.3f%% %10.3f%% %10.3f%%' % ('SimpleTagBased', n, stb, re_stb, pre_stb, coverage, stb_diversity, stb_aver))
            print('%3s %12d %10s %10.3f%% %10.3f%% %10.3f%% %10.3f%% %10.3f%%' % ('NormTagBased', n, ntb, re_ntb, pre_ntb, coverage, ntb_diversity, ntb_aver))
            print('%3s %10d %10s %10.3f%% %10.3f%% %10.3f%% %10.3f%% %10.3f%%' % ('TagBased-TFIDF', n, tfidf, re_tfidf, pre_tfidf, coverage, tfidf_diversity, tfidf_aver))
            print('\n')

    def prediction_main(self):
        self.loadData()
        self.randomlySplitData()
        self.initStat()
        self.testRecommend()


if __name__ == '__main__':
    NTB = DeliciousRecommend()
    NTB.prediction_main()



