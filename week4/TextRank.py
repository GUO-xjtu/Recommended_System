import re
import jieba
from jieba import analyse
import numpy as np
import jieba.posseg as pseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from gensim.summarization.summarizer import summarize


class TRAnalyze(object):
    def __init__(self):
        f = open(r'../week4/data/news.txt', 'r', encoding='gbk')
        self.news = f.read()

    def tr4_key_word(self):
        """
        三种分词模式：words_no_filter 简单分词，不进行词性过滤，不剔除停用词
                    words_no_stop_words 剔除停用词
                    words_all_filters 剔除停用词，也进行词性过滤
        :return:
        """
        tr4w = TextRank4Keyword()
        # 获取关键字，对出项的英文进行小写，窗口为2
        tr4w.analyze(text=self.news, lower=True, window=2)
        # 提取前20位关键词，每个关键词长度至少为1
        for item in tr4w.get_keywords(20, word_min_len=1):
            print(item)
        print('\n\n')

    def jie_key_word(self):
        # 获取分词
        word_list = jieba.cut(self.news, cut_all=False)
        print(' '.join(word_list))
        # 获取分词和词性
        # words = pseg.cut(self.news)
        # for item in words:
        #     print(item)
        # 使用TextRank算法获取关键字
        keywords = analyse.textrank(self.news, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
        print(keywords)
        # 使用TF-IDF获取关键字
        keywords = analyse.extract_tags(self.news, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
        print(keywords)

    def sk_key_words(self):
        r = '[’!"#$%&\'()*+,-.。，《》：“”\"/:;<=>?@[\\]^_`{|}~]+'
        self.news = self.news.split('。')
        news = []
        for i in self.news:
            new = re.sub(r, '', i)
            new = ' '.join(jieba.cut(sentence=new, cut_all=False))
            news.append(new)

        print(news)
        # 将文本中的词语转换为词频矩阵，a[i][j]指j词在i类文本下的词频
        vectorizer = CountVectorizer()

        # 统计每个词的TF-IDF权值
        transformer = TfidfTransformer()
        # 将文本转化为词频矩阵
        word = vectorizer.fit_transform(news)
        # 计算tf-idf
        word_tfidf = transformer.fit_transform(word)
        # 获取词袋模型中的词语
        word = vectorizer.get_feature_names()
        print(word)
        # 抽取出tf-idf矩阵，获取权重
        weight = word_tfidf.toarray()
        print(weight)
        for i in range(len(weight)):
            print(list(zip(word, weight[i])))

    def tr4_key_sentence(self):
        tr4s = TextRank4Sentence()
        tr4s.analyze(text=self.news, lower=True, source='all_filters')
        for item in tr4s.get_key_sentences(num=3):
            print(item)

    def gen_key_sentence(self):
        """
        gensim模块的summarization使用的也是TextRank算法，不过其的文档预处理方式是按照英文进行的，
        本方法先用jieba分词预处理为gensim可接受的方式。

        :return:
        """
        word = jieba.cut(self.news)
        word = ' '.join(word)
        sentence = summarize(word)
        sentence = sentence.replace(' ', '')
        print(sentence)
if __name__ == '__main__':
    TRA = TRAnalyze()
    #TRA.tr4_key_word()
    # TRA.jie_key_word()
    # TRA.sk_key_words()
    TRA.tr4_key_sentence()
    TRA.gen_key_sentence()
