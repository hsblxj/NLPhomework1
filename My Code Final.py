import os
import re
import math
import jieba
from collections import Counter


def Pretreatment():
    # 读文件
    txt_path = r".\jyxstxtqj_downcc.com"
    txtlist = os.listdir(txt_path)
    pathlist = []
    for name in txtlist:
        if name.lower().endswith('.txt'):
            path = os.path.join(txt_path, name)
            pathlist.append(path)

    # 存储语料，按自然段分割
    content = []
    for path in pathlist:
        with open(path, "r", encoding="ANSI") as file:
            text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
            content += text

    # 返回文本
    return content


def One_Gram(txt2):
    # 去除噪音，删去非汉字字符、空格、段落
    content = txt2
    chinese_str = ".*?([^\u4E00-\u9FA5]).*?"  # 匹配中文字符的正则表达式
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 特殊符号
    for j in range(len(content)):
        content[j] = re.sub(english, "", content[j])
        content[j] = re.sub(chinese_str, '', content[j])
        content[j] = content[j].replace("\n", '')
        content[j] = content[j].replace(" ", '')
        content[j] = content[j].replace(u'\u3000', '')
    print("1-gram:")
    token = []
    for para in content:
        token += jieba.lcut(para)
    # N_Gram求解
    # 计算词频并排序
    word_freq = dict()  # 空字典
    for word in token:
        if word not in word_freq:
            word_freq[word] = 1
        word_freq[word] += 1
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # 计算信息熵
    word_tf = dict()
    shannoEnt = 0.0
    for word, freq in word_freq:
        # 计算p(xi)
        prob = freq / len(token)
        word_tf[word] = prob
        shannoEnt -= prob * math.log(prob, 2)
    # 输出前10及词频并输出信息熵
    print("字库总字数：", len(token), "\n", "不同字的个数：", len(word_freq))
    print("entropy_1gram:", shannoEnt)
    print("出现频率前10的字：", word_freq[:10])


def Two_Gram(txt3):
    # 去除噪音，删去非汉字字符、空格、段落
    content = ''
    for j in range(len(txt3)):
        content += txt3[j]
    jieba.load_userdict("cn_stopwords.txt")  # 导入停词词库
    seg_list = jieba.lcut_for_search(content)
    for j in range(len(seg_list)):
        rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
        seg_list[j] = rule.sub('', seg_list[j])
    seg_list = [i for i in seg_list if i != '']
    seg_list = [i for i in seg_list if len(i) != 1]
    # N_Gram求解
    print("2-gram:\n")
    # 计算字频并排序
    word_freq = Counter(seg_list)
    ansdict = word_freq.most_common()  # 返回list

    # 计算信息熵
    word_freq = ansdict

    # 构造仅有首词的list
    first_word = []
    for j in range(len(seg_list)):
        first_word.append(seg_list[j][0])
    firstword_freq = Counter(first_word)
    firstword_ansdict = firstword_freq.most_common()

    shannoEnt = 0.0
    for j in range(len(word_freq)):
        prob = word_freq[j][1] / len(seg_list)  # 联合概率P(xy)
        temp_times = 0
        for k in range(len(firstword_ansdict)):
            if firstword_ansdict[k][0][0] == word_freq[j][0][0]:
                temp_times += firstword_ansdict[k][1]
        # freq/temp_times 条件概率P(x|y)
        shannoEnt -= prob * math.log(word_freq[j][1]/temp_times, 2)
    # 输出前10及词频并输出信息熵
    print("词库总词数：", len(seg_list), "\n", "不同词的个数：", len(word_freq))
    print("entropy_2gram:", shannoEnt)
    print("出现频率前10的词组：", word_freq[:10])


if __name__ == "__main__":
    # 预处理，将乱码信息去除，返回处理后文本
    txt1 = Pretreatment()
    # 1-Gram
    One_Gram(txt1)
    # 2-Gram
    Two_Gram(txt1)
