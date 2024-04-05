# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/1  21:49
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------
import time

import torch
from config.main_config import dataset_config, train_config
from dataset.SaleDataset import SaleDataset
from models.SaleModel import SaleModel
from utils.model_tool import save_model


def train_tool():
    device = torch.device('cpu')
    # 判断是否保存
    is_save = 0
    with open("is_save.txt", "w") as f:
        f.write(str(is_save))
    # 1、创建数据集
    train_dataset = SaleDataset(dataset_config['train_path'])
    # 2、配置数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True,
                                                   num_workers=2)
    train_dataset_size = len(train_dataloader)

    # 3、实例化模型，损失函数，优化器
    model = SaleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    # 如果有训练的模型，则进行加载
    if train_config["model_load_path"]:
        checkpoint = torch.load(train_config["model_load_path"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("载入{}作为网络模型".format(train_config["model_load_path"]))
    else:
        print("没有加载模型，路径为空")
    loss_fun = torch.nn.L1Loss()

    # 4、训练模型
    model = model.to(device)
    model.train()
    for epoch in range(1, train_config["epoch"] + 1):
        for iteration, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            time.sleep(0.5)
            # 开始梯度归零，反向传播，参数更新
            optimizer.zero_grad()
            loss = loss_fun(out, y)
            loss.backward()
            optimizer.step()
            if iteration % train_config["show_iteration"] == 0:
                print("Iter:{}/{} Loss:{}".format(iteration, train_dataset_size,round( loss.item(), 4)))
                # print("out:{}  y:{}".format(out,y))
            # 进行判断保保存
            with open("is_save.txt", "r") as f:
                is_save = int(f.read())
            if is_save:
                print("进行保存!!  保存后停止")
                save_model(model, optimizer, train_config["model_save_path"] + str(epoch) + "_sale" + ".pth")
                return
        save_model(model, optimizer, train_config["model_save_path"] + str(epoch) + "_sale" + ".pth")
