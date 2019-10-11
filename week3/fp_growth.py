
class TreeNode(object):
    """
    创建FP树的类
    name:节点元素名称，在构造时初始化为给定值
    count:出现的次数，构造时初始化为给定值
    nodeLink:指向下一个相似节点的指针，默认为None
    parent:指向父节点的指针，在构造时初始化为给定值
    children:指向字节点的字典，以子节点的元素名称为键，指向子节点的指针为值，初始化为空
    node_num():增加节点的出现次数
    put_node_tree():输出节点和子节点的FP树结构
    """
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    # 增加节点的出现次数
    def node_num(self, numOccur):
        self.count += numOccur

    # 输出节点和子节点的FP树结构
    def put_node_tree(self, ind=1):
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.put_node_tree(ind+1)


# 初始化数据集
def create_init_set(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


# 创建FP树
def create_tree(retDict, minSup=2):
    """
    先统计数据集汇总各个元素出现的频数， 将频数小于minSup的元素删除，然后将数据集中各条记录按出现的频数排序，
    剩下的元素为频繁项。
    用更新后的数据集中的每条记录构建FP树，同时更新头指针表
    :param dataSet: 初始数据集
    :param minSup: 最小支持度，默认为2
    :return: 返回的是构建的FP树和头指针表
    """
    headerTable = {}  # 第一次遍历数据集，创建头指针表
    for trans in retDict:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + retDict[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    # 空元素集，返回空
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = TreeNode('Null Set', 1, None)  # 跟节点

    # 第二次遍历数据集，创建FP树
    for transSet, count in retDict.items():
        localID = {}  # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in transSet:
            if item in freqItemSet:
                localID[item] = headerTable[item][0]  # 这个0是之前加入过一个数据项
        if len(localID) > 0:
            # 按照全局频数从大到小，对单样本排
            orderedItems = [v[0] for v in sorted(localID.items(), key=lambda p: p[1], reverse=True)]
            update_tree(orderedItems, retTree, headerTable, count)  # 更新FP树
    return retTree, headerTable


# 更新FP树
def update_tree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 有该元素时，计数值加1
        inTree.children[items[0]].node_num(count)
    else:
        # 没有这个元素时创建一个新节点
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
        # 更新头指针表或前一个相似元素项节点的指针指向新节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            update_header(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:
        # 对剩下的元素项迭代调用updata_tree函数
        update_tree(items[1::], inTree.children[items[0]], headerTable, count)


# 获取头指针表中该元素项对应的单链表的尾节点，然后将其指向新节点targetNode
def update_header(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


# 创建前缀路径
def find_prefix_path(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascend_tree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def ascend_tree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascend_tree(leafNode.parent, prefixPath)


# 查找频繁项集
def mine_tree(headerTable, minSup, preFix, freqItemList):
    """
    针对每一个频繁项，逐个挖掘频繁项集。
    先获取频繁项的前缀路径，将前缀路径作为新的数据集，以此构建前缀路径的条件FP树，
    然后对条件FP树中的每一个频繁项，获取前缀路径并以此构建新的条件路径，不断迭代，
    直到条件FP树只包含一个频繁项。
    :param headerTable: 头指针表
    :param minSup: 最小支持度
    :param preFix: 频繁项，初始为空集合
    :param freqItemList:频繁项集，初始为空字典:{频繁项集：出现次数}
    :return:
    """
    # 按照物品出现次数，从小到大排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]
    if len(bigL) == 0:
        return
    # print('频繁项从小到大排序为：', bigL)
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        support = headerTable[basePat][0]
        freqItemList[frozenset(newFreqSet)] = support
        condPattBases = find_prefix_path(basePat, headerTable[basePat][1])  # 获取前缀路径
        myCondTree, myHead = create_tree(condPattBases, minSup)

        if myHead is not None:
            # print('basePat: ', basePat)
            # print('conditional tree for: ', newFreqSet)
            # myCondTree.put_node_tree()
            # print('\n\n\n')
            mine_tree(myHead, minSup, newFreqSet, freqItemList)


# 移除需评判元素
def remove_str(set, str):
    tempSet = []
    for elem in set:
        if elem != str:
            tempSet.append(elem)
    tempFrozenSet = frozenset(tempSet)
    return tempFrozenSet


# 挖掘关联规则
def rules_generator(freqItems, minConf, rules):
    for items in freqItems:
        if len(items) > 1:
            get_rules(items, items, rules, freqItems, minConf)


def get_rules(freqset, currset, rules, freqItems, minConf):
    for item in currset:
        subSet = remove_str(currset, item)
        # print('item: ', item)
        # print('currset: ', currset)
        # print(currset, '： ', freqItems[freqset])
        # print(subSet, '：', freqItems[subSet])
        confidence = freqItems[freqset] / freqItems[subSet]
        # print('confidence: ', confidence)
        # print((subSet, freqset-subSet, confidence))
        # print('\n\n')
        # 若置信度大与阈值，则保存此关联规则
        if confidence >= minConf:
            flag = False
            for rule in rules:
                if rule[0] == subSet and rule[1] == (freqset - subSet):
                    flag = True
            if flag is False:
                rules.append((subSet, freqset-subSet, confidence))
            if len(subSet) >= 2:
                get_rules(freqset, subSet, rules, freqItems, minConf)


# 封装
def fpGrowth(dataSet, minSup=3, minConf=0.5):
    initSet = create_init_set(dataSet)
    print('数据集初始化：', initSet)
    myFPtree, myHeaderTab = create_tree(initSet, minSup)
    freqItems = {}
    print('头指针表：', myHeaderTab)
    mine_tree(myHeaderTab, minSup, set([]), freqItems)
    rules = []
    rules_generator(freqItems, minConf, rules)
    print(freqItems.items())
    freqItems = [item for item in sorted(freqItems.items(), key=lambda b:b[1], reverse=True)]
    rules = [item for item in sorted(rules, key=lambda b: b[2], reverse=True)]
    return freqItems, rules


# 测试
def load():
    simDit = [['r', 'z','h','j','p'],['z','y','x','w','v','u','t','s'],
              ['z'], ['r','x','n','o','s'], ['y','r','x','z','q','t','p'],
              ['y','z','x','e','q','s','t','m']
    ]
    return simDit

# initdata = load()
# initset = create_init_set(initdata)
# # print(initset)
# myFptree, myHeader = create_tree(initset, 2)
# condpats = find_prefix_path('t', myHeader['t'][1])
# # print(myFptree.put_node_tree())
# # print(myHeader)
# # print('t:', condpats)
# frm = {}
# mine_tree(myHeader, 2, set(), frm)
# print(frm)
# rules = []
# rules_generator(frm, 0.2, rules)
# print(rules)