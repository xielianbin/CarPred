# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/1  21:39
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------
import time

import torch

from config.main_config import dataset_config, predict_config
from dataset.SaleDataset import SaleDataset
from models.SaleModel import SaleModel


def predict_tool():
    device = torch.device('cpu')

    ## 1、创建数据集
    predict_dataset = SaleDataset(dataset_config['predict_path'])
    # 2、配置数据加载器
    predict_dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=predict_config["batch_size"], shuffle=False,
                                                   num_workers=2)
    predict_dataset_size = len(predict_dataloader)

    # 3、实例化模型，损失函数，优化器
    model = SaleModel()
    # 如果有训练的模型，则进行加载
    if predict_config["model_predict_path"]:
        checkpoint = torch.load(predict_config["model_predict_path"])
        model.load_state_dict(checkpoint["model"])
        print("载入{}作为网络模型".format(predict_config["model_predict_path"]))
    else:
        print("没有加载模型，路径为空")
        return

    # 模型开启验证，输出预测的数据
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        # truth_img_path_list参数是用来保存遥感影像的路径
        for iteration, (x, y) in enumerate(predict_dataloader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            time.sleep(0.5)
            out=out.numpy()*100000
            y=y.numpy()*100000
            print("Iter:{}/{} ".format(iteration, predict_dataset_size))
            # 保存预测的结果，保存结果，需要乘以缩放尺寸，在dataset_config.yaml配置文件里面
            print("out:{} y:{}".format(out,y))
            # save_img(model_out_data_val*10000,predict_config["predict_dataset_dir"],predict_config["model_output_path"],truth_img_path_list)

    print("预测结束！")