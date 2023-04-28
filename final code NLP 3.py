import os
import random
import re
from collections import defaultdict
import jieba
import numpy as np
from gensim import corpora, models
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def pretreatment():
    # 读取所有 .txt 文件的路径
    txt_path = r".\jyxstxtqj_downcc.com"
    txtlist = [os.path.join(txt_path, name) for name in os.listdir(txt_path) if name.lower().endswith('.txt')]
    print(txtlist)

    # 用正则表达式和 jieba 分词对文本进行预处理
    ad = '[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    jieba.load_userdict("cn_stopwords.txt")
    text_dict = {}
    for path in txtlist:
        if path.split('.')[-1] == 'txt' and path != '.\\jyxstxtqj_downcc.com\\inf.txt':
            with open(path, "r", encoding='ansi') as file:
                file_content = file.read()
            file_content = file_content.replace("本书来自www.cr173.com免费txt小说下载站", '').replace("更多更新免费电子书请关注www.cr173.com",
                                                                                           '')
            file_content = re.sub(ad, '', file_content).replace("\n", '').replace(" ", '').replace('\u3000', '')

            new_words_lst = [word for word in jieba.cut(file_content)]
            text_dict[os.path.splitext(os.path.basename(path))[0]] = new_words_lst
            print(os.path.splitext(os.path.basename(path))[0], '总词数：', len(new_words_lst))
    return text_dict


def base_on_word(text_dict, paragraph_num, paragraph_length, num_topics, random_consistent=True):
    text_list = []  # 用于保存所有的段落
    label_list = []  # 用于保存所有段落的标签
    for label, text in text_dict.items():
        # 对于每篇文章，随机抽取一定数量、一定长度的段落
        for i in range(paragraph_num):
            # 将段落的标签添加到标签列表中
            label_list.append(label)
            # 随机抽取一个起始位置，然后取出指定长度的段落
            random_int = random.randint(0, len(text) - paragraph_length - 1)
            paragraph = text[random_int:random_int + paragraph_length]
            text_list.append(paragraph)

    # 将标签列表从字符串形式映射为整数形式，并保证标签与文本列表长度相同
    label_dict = defaultdict(int)  # 定义一个默认字典
    for label in label_list:
        label_dict[label] += 1  # 统计每个标签出现的次数
    label_dict = {label: idx for idx, label in enumerate(label_dict.keys())}  # 将标签映射为整数
    int_label_list = [label_dict[label] for label in label_list]  # 将标签列表从字符串形式转换为整数形式
    assert len(int_label_list) == len(text_list)  # 确保标签列表与文本列表长度相同

    # 使用索引将标签列表和数据列表的顺序按照相同方式打乱
    random_seed = 10
    if random_consistent:
        np.random.seed(random_seed)  # 设置随机种子以保证每次运行程序时的结果相同
    indices = np.random.permutation(len(int_label_list))  # 生成一个打乱顺序的索引
    int_label_list = [int_label_list[i] for i in indices]  # 使用索引将标签列表打乱顺序
    text_list = [text_list[i] for i in indices]  # 使用索引将数据列表打乱顺序

    # 划分训练集和测试集
    train_p = 0.6  # 训练集所占比例
    train_num = int(len(text_list) * train_p)  # 训练集数量
    label_train = int_label_list[:train_num]  # 训练集标签
    label_test = int_label_list[train_num:]  # 测试集标签
    text_train = text_list[:train_num]  # 训练集文本
    text_test = text_list[train_num:]  # 测试集文本

    # 将文本列表转化为LDA模型需要的输入——文本向量
    dictionary = corpora.Dictionary(text_list)  # 构建词典
    corpus_train = [dictionary.doc2bow(doc) for doc in text_train]  # 训练集向量表示
    corpus_test = [dictionary.doc2bow(doc) for doc in text_test]  # 测试集向量表示

    # 训练LDA模型
    lda = models.LdaModel(corpus=corpus_train, id2word=dictionary, num_topics=num_topics)  # 训练LDA模型

    # 获取训练集和测试集的每个段落的主题分布
    topics_train = lda.get_document_topics(corpus_train, minimum_probability=0)  # 训练集主题分布
    topics_test = lda.get_document_topics(corpus_test, minimum_probability=0)  # 测试集主题分布

    feature_train = []  # 初始化训练集特征向量
    feature_test = []  # 初始化测试集特征向量

    for i in range(0, len(topics_train)):
        feature_train.append([k[1] for k in topics_train[i]])

    for i in range(0, len(topics_test)):
        feature_test.append([k[1] for k in topics_test[i]])

    # print('训练集特征矩阵大小为:', np.array(feature_train).shape)
    # print('测试集特征矩阵大小为:', np.array(feature_test).shape)

    # 定义SVM分类器
    clf = SVC(kernel='rbf', decision_function_shape='ovr')

    # 训练模型
    clf.fit(feature_train, label_train)

    # 测试模型
    predict_train = clf.predict(feature_train)
    predict_test = clf.predict(feature_test)

    # print(list(predict_test))
    # print(label_test)
    # accuracy_train = clf.score(feature_train, label_train)
    # print(f'Train Accuracy: {100 * accuracy_train:.2f}%')
    accuracy_test = clf.score(feature_test, label_test)
    # print(f'Test Accuracy: {100 * accuracy_test:.2f}%')

    return accuracy_test


def base_on_char(text_dict, paragraph_num, paragraph_length, num_topics):
    for label, text in text_dict.items():
        text_dict[label] = list(''.join(text))
    return base_on_word(text_dict, paragraph_num, paragraph_length, num_topics, random_consistent=True)


if __name__ == "__main__":
    # 前处理，读取文档，标签为所属小说,均匀抽取200个段落
    txt1_dict = pretreatment()
    num_topics = [i for i in range(1, 101, 5)]
    accuracy_test_word = []
    accuracy_test_char = []
    # 字基本单元与词基本单元
    for num1 in num_topics:
        print(num1)
        accuracy_test_word.append(base_on_word(txt1_dict, paragraph_num=200, paragraph_length=800, num_topics=num1))
        accuracy_test_char.append(base_on_char(txt1_dict, paragraph_num=200, paragraph_length=800, num_topics=num1))
    print(accuracy_test_word)
    print(accuracy_test_char)
    print(num_topics)
    plt.plot(num_topics, accuracy_test_word, label='Accuracy Test Word')
    plt.plot(num_topics, accuracy_test_char, label='Accuracy Test Char')
    plt.xlabel('Num Topics')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Test Word/Char by Num Topics')
    plt.legend()
    plt.show()
