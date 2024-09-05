# 贝叶斯定理：先验、后验、条件概率； 对于待分类的项，求解各个类出现的概率，哪个概率最大就属于哪个类
# 朴素贝叶斯实现垃圾邮件分类
import numpy as np
import re
import random

'''
函数说明：将切分的实验样本词条整理成不重复的词条列表
: param dataset 样本数据集
: return vocabset 词汇表
'''
def createVocabList(dataset):
    vocabset = set()  # 定义一个空集合
    for document in dataset:
        vocabset = vocabset | set(document)  # 去并集
    return list(vocabset) 

'''
函数说明：根据vocablist 词汇表将inputset 向量化，向量的每个元素为1或0
: param vocablist 返回的词汇表
: param inputset 切分的词条列表
: return returnvec 文档向量，词条模型
'''

def setofwords2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[returnvec.index(word)] = 1
        else: 
            print(f"the word: {word} is not in my vocalbulary! ")
    return returnvec  


'''词袋模型'''
def setofwords2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[returnvec.index(word)] += 1
    return returnvec

"""
朴素贝叶斯分类器训练函数
: param trainmatrix 训练文档矩阵
: param traincategory 训练类别标签向量
: return
    p0vect 正常邮件类的条件概率组
    p1vect 垃圾邮件类的条件概率组
    pabusive 文档属于垃圾邮件类的概率
"""

def trainNB0(trainmatrix,traincategory):
    numtraindocs = len(trainmatrix)
    numwords = len(trainmatrix[0])
    pabusive = sum(traincategory)/float(numtraindocs)
    p0num = np.ones(numwords)
    p1num = np.ones(numwords)

    # 使用Laplace平滑处理方法解决零概率问题
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numtraindocs):
        if traincategory[i] == 1: # 统计属于侮辱类的条件概率所需的数据
            p1num += trainmatrix[i]
            p1Denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0Denom += sum(trainmatrix[i])
    p1vect = np.log(p1num/p1Denom)
    p0vect = np.log(p0num/p0Denom)
    return p0vect, p1vect, pabusive


"""
朴素贝叶斯分类器分类函数
: param
    vec2classify 待分类的词条数组
    p0vec 正常邮件类的条件概率数组
    p1vec 垃圾邮件类的条件概率数组
    pclass1 文档属于垃圾邮件的概率
: return 
    0 属于正常邮件类
    1 属于异常邮件类
"""

def classifyNB(vec2classify, p0vec, p1vec,pclass1):
    p1 = sum(vec2classify * p1vec) + np.log(pclass1)
    p0 = sum(vec2classify * p0vec) + np.log(1.0-pclass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
"""
接收一个大字符串并将其解析为字符串列表
"""
def textParse(bigString):
    listofTokes = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listofTokes if len(tok) > 2]