# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/2  9:46
# @Email    : 2826389624@qq.com
# @Function : EDA 数据探索分析
# -------------------------
from matplotlib import pyplot as plt

from config.main_config import config
import seaborn as sns

def continuous_EDA(df):
    # 连续特征
    print("统计连续特征")
    for col in config['num_cols']:
        # 绘制密度图
        sns.kdeplot(df[col], fill=True)
        # 设置图形标题和标签
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Density')

        plt.show()
def scatter_EDA(df):
    print("统计离散特征")
    # 离散特征
    for col in config['cate_cols']:
        # 统计特征频次
        counts = df[col].value_counts()

        # 绘制条形图
        counts.plot(kind='bar')

        # 设置图形标题和标签
        plt.title(f'{col} Frequencies')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # 显示图形
        plt.show()