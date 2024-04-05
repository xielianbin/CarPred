# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/1  21:38
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------

# 训练模型，验证模型，这里就是八股文，熟悉基础pipeline
def train_model(model, train_loader, optimizer, device, metric_list=['mean_absolute_error']):
    model.train()
    pred_list = []
    label_list = []
    max_iter = int(train_loader.dataset.__len__() / train_loader.batch_size)
    for idx, data in enumerate(train_loader):
        # 把数据拷贝在指定的device
        for key in data.keys():
            data[key] = data[key].to(device)
        # 模型前向+Loss计算
        output = model(data)
        pred = output['pred']
        loss = output['loss']
        # 八股文完成模型权重更新
        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['price'].squeeze(-1).cpu().detach().numpy())

        if idx % 50 == 0:
            logger.info(f"Iter:{idx}/{max_iter} Loss:{round(loss.item(), 4)}")

    res_dict = dict()
    for metric in metric_list:
        res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict
