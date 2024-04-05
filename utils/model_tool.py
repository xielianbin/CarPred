# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/4  21:04
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------
from torch import save
def save_model(model,optimizer,save_path):
    state={"model":model.state_dict(),"optimizer":optimizer.state_dict()}
    save(state,save_path)