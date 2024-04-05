# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/1  21:38
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------

def valid_model(model, valid_loader, device, metric_list=['mean_absolute_error']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (valid_loader):
        # 把数据拷贝在指定的device
        for key in data.keys():
            data[key] = data[key].to(device)
        # 模型前向
        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['price'].squeeze(-1).cpu().detach().numpy())

    res_dict = dict()
    for metric in metric_list:
        res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict

