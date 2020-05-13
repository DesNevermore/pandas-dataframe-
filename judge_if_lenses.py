# coding: utf-8

import pandas as pd
import math
import numpy

def load_data():
    # 读取文件，dataframe中列分别为年龄，视力缺陷，散光情况，泪腺分泌情况，需配眼镜情况
    df = pd.read_csv('lenses.txt', sep='	', names=['age', 'eye_defect', 'astigmatism', 'lacrimal', 'lenses'])
    return df

# 计算信息增益,返回选择分类属性
def info_gain(df ,classify):
    attr_dic = {}
    # 首先计算决策属性的信息熵
    print(df.lenses.value_counts())
    c = []
    for item in classify['lenses']:
        c.append(df.lenses.value_counts().get(item, 0))
    # 计算决策属性信息熵hd
    total = 0
    for item in range(3):
        total += c[item]
    print(total)
    hd = 0
    for i in range(3):
        if c[i] == 0:
            continue
        else:
            hd += -(abs(c[i])/total) * (math.log(abs(c[i])/total))
    print(hd)
    # 计算各列属性的信息熵
    for attr in df.columns[:df.shape[1]-1]:
        c1 = classify[attr]
        print(c1)
        # print(df[attr].value_counts())
        # 计算此列中每个值的数量
        D = []
        for item in classify[attr]:
            D.append(df[attr].value_counts().get(item))
        print(D, '\n')
        # 计算Dik
        HD = []
        for i in range(len(c1)):
            Dik = []
            for j in classify['lenses']:
                Dik.append(df.loc[df[attr] == c1[i]]['lenses'].value_counts().get(j, 0))    # 后面的0是默认值，不然他就会返回None
            # print(type(Dik[1]))
            # Dik的是当前这个属性所对应的lenses的标签
            print(Dik)
            x = 0
            for y in range(3):
                if Dik[y] == 0:
                    continue
                else:
                    x += -(Dik[y]/D[i]) * (math.log(Dik[y]/D[i]))
            HD.append(x)
        print(HD)
        # 计算各个列信息增益
        g = 0
        for u in range(len(c1)):
            g = (D[u]/total) * HD[i]
        gDA = hd - g
        print('信息增益：', gDA)
        attr_dic[attr] = gDA
        print('\n\n')
    print(attr_dic)
    return dic_max(attr_dic)

def dic_max(dic):
    return sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]

# 返回某一属性的某一种value的dataframe
def split_data(df, attr, value):
    # print(df)
    return df.loc[df[attr] == value]

def createTree(df, classify):
    if len(df) == 1:
        return df.iat[0, 0]
    class_attr = info_gain(df, classify)
    # print(class_attr)
    myTree = {class_attr: {}}
    # print(df)
    # df.drop([class_attr], axis=1, inplace=True)
    # print(df)
    # print(myTree)

    # print(gf)
    # 开始分枝
    for value in classify[class_attr]:
        # 根据当前的class_attr的值，划分数据集
        hf = split_data(df, class_attr, value)
        print(hf)
        # 删除当前列
        # hf.drop([class_attr], axis=1, inplace=True)
        gf = hf.drop([class_attr], axis=1)
        myTree[class_attr][value] = createTree(gf, classify)
    return myTree

def catagory(inputTree, testAttr):
    firststr = list(inputTree.keys())[0]
    second_dict = inputTree[firststr]

    for key in second_dict.keys():
        if testAttr[firststr] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = catagory(second_dict[key], testAttr)
            else:
                class_label = second_dict[key]
    return class_label

if __name__ == '__main__':
    classify = {'age': ['young', 'pre', 'presbyopic'],
                'eye_defect': ['myope', 'hyper'],
                'astigmatism': ['yes', 'no'],
                'lacrimal': ['reduced', 'normal'],
                'lenses':['no lenses', 'soft', 'hard']
                }
    # 读文件，生成dataframe框架
    df = load_data()
    # 划分训练集生成决策树
    # print(df.loc[0, 'lenses'])
    # print(df.loc[df['age'] == 'young']['lenses'].value_counts())
    myTree = createTree(df, classify)
    print(myTree)
    testAttr = {'age': 'young',
                'eye_defect': 'myope',
                'astigmatism': 'no',
                'lacrimal': 'reduced'}

    # print(type(myTree))
    # print(list(myTree.keys())[0])
    label = catagory(myTree, testAttr)
    print(label)