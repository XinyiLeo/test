# -*- coding: utf-8 -*-

import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np

#解决matplotlib中文显示问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 输入是一DataFrame，每一列是一支股票在每一日的价格
def find_cointegrated_pairs(dataframe):
    # 得到DataFrame长度
    n = dataframe.shape[1]
    # 初始化p值矩阵
    pvalue_matrix = np.ones((n, n))
    # 抽取列的名称
    keys = dataframe.keys()
    # 初始化强协整组
    pairs = []
    # 对于每一个i
    for i in range(n):
        # 对于大于i的j
        for j in range(i+1, n):
            # 获取相应的两只股票的价格Series
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            # 分析它们的协整关系
            result = sm.tsa.stattools.coint(stock1, stock2)
            # 取出并记录p值
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            # 如果p值小于0.05
            if pvalue < 0.05:
                # 记录股票对和相应的p值
                pairs.append((keys[i], keys[j], pvalue))
    # 返回结果
    return pvalue_matrix, pairs

# 输入是一DataFrame，每一列是日数据，
def find_OLS_pairs(dataframe):
    # 得到DataFrame长度
    n = dataframe.shape[1]
    # 初始化p值矩阵
    pvalue_matrix = np.ones((n, n))
    # 抽取列的名称
    keys = dataframe.keys()
    # 初始化强协整组
    pairs = []
    # 对于每一个i
    for i in range(n):
        # 对于大于i的j
        for j in range(i+1, n):
            # 获取相应的两只股票的价格Series
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            plt.plot(stock1)
            plt.plot(stock2)
            plt.xlabel("时间")
            plt.ylabel("价格")
            plt.legend([keys[i], keys[j]], loc='best')
            #plt.savefig(str(i) + str(j) +"price.png")

            #OLS拟合
            x = stock1
            y = stock2
            X = sm.add_constant(x)
            result = (sm.OLS(y, X)).fit()
            print(result.summary())
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x, y, 'o', label="data")
            ax.plot(x, result.fittedvalues, 'r', label="OLS")
            ax.legend(loc='best')
            plt.title(keys[i]+keys[j])

            #plt.savefig(str(i) + str(j) + "OLS.png")
            plt.show()

#文件路径
industryFilePath = "D:\\inputData\\industry.xls"
rbFilePath = "D:\\inputData\\rb.xls"
hcFilePath = "D:\\inputData\\hc.xls"

#获取数据
industryDf = pd.read_excel(industryFilePath, sheetname = 0, skiprows=[0], index_col=0).dropna()
rbDf = pd.read_excel(rbFilePath, sheetname = 0, skiprows=[1], index_col=0).dropna()
hcDf = pd.read_excel(hcFilePath, sheetname = 0, skiprows=[1], index_col=0).dropna()

inputData = pd.concat([industryDf, rbDf.loc[industryDf.index], hcDf.loc[industryDf.index]], axis=1).dropna()
print(inputData)

#获得价差
# df['diff'] = df.iloc[:,1] - df.iloc[:,2]
# priceDiff = pd.Series(df['diff'])


#价差画图
# plt.plot(priceDiff)
# ax=plt.gca()
# ax.set_xticklabels( pd.Series(df.iloc[:,0]))
# plt.show()

#输出相关性
print((inputData.corr()))

#回归分析
find_OLS_pairs(inputData)

#协整性分析
# pvalues, pairs = find_cointegrated_pairs(inputData)
# sns.heatmap(1-pvalues, xticklabels=inputData.columns.values , yticklabels=inputData.columns.values, cmap='RdYlGn_r', mask = (pvalues == 1))
# print(pairs, pvalues)

#写入excel
writer = pd.ExcelWriter('D:\\inputData\\output.xlsx')
(inputData.corr()).to_excel(writer,'Sheet1')
writer.save()


