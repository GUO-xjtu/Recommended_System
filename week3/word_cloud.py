import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class WCloud(object):
    def __init__(self):
        self.file = '../week3/data/Market_Basket_Optimisation.csv'
        self.my_text = None

    # 单词切分，去掉标点
    def cut_word(self):
        with open(self.file) as f:
            mytext = f.read()
        mytext = mytext.replace(',', ' ')
        self.my_text = " ".join(jieba.cut(mytext))
        print(self.my_text)

    # 生成词云
    def cloud(self):
        wordcloud = WordCloud().generate(self.my_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    WC = WCloud()
    WC.cut_word()
    WC.cloud()
