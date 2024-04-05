# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/2  9:38
# @Email    : 2826389624@qq.com
# @Function : 在加载数据集的时候
# 1、EDA数据探索操作
# 2、简易特征编码，处理空值异常值
# -------------------------

import pandas as pd
from config.main_config import dataset_config
from torch import Tensor
class SaleDataset:
    def __init__(self, dataset_path):
        # self.state=state
        self.df = pd.read_csv(dataset_path, sep=' ')
        # 输出数据集df的信息
        # print(df.info())
        # fillna是填充空值
        for col in dataset_config['dataset_cols']:
            self.df[col] = self.df[col].fillna(0)

    def __getitem__(self, index):
        x = []
        for col in dataset_config['dataset_cols']:

            try:
                temp = float(self.df[col].iloc[index])

            except Exception as e:
                # print(e)
                temp=0.
            x.append(temp / dataset_config["dataset_scale"][col])

        x=Tensor(x)
        # if self.state=="train":
        y=self.df["price"].iloc[index]/ dataset_config["dataset_scale"]["price"]
        y=float(y)
        y=Tensor([y])
        # print(x,y)
        return x,y
        # else:
        #     return x

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    dt = SaleDataset(dataset_config['train_path'],"train")
    for i in range(100):
        print(len(dt.__getitem__(i)))
