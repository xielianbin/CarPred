# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/2  9:26
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------
from torch.nn import Module,Linear,ReLU

class SaleModel(Module):
    def __init__(self):
        super(SaleModel,self).__init__()
        self.l1=Linear(29,32)
        self.act1=ReLU(True)
        self.l2=Linear(32,64)
        self.act2=ReLU(True)
        self.l3=Linear(64,32)
        self.act3=ReLU(True)
        self.l4=Linear(32,1)

    def forward(self, x):
        out=self.act1(self.l1(x))
        out=self.act2(self.l2(out))
        out=self.act3(self.l3(out))
        out=self.l4(out)
        return out
