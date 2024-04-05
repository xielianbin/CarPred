# -*- coding: utf-8 -*-
# -------------------------
# @Author   : xielianbin
# @Time     : 2024/4/1  21:45
# @Email    : 2826389624@qq.com
# @Function :
# -------------------------
dataset_config = {
    "train_path": './asset/used_car_train_20200313.csv',
    "predict_path": './asset/used_car_train_20200313.csv',
    # "predict_path": './asset/used_car_testB_20200421.csv',
    "dataset_cols": ["name", "regDate",
                     "model", "brand", "bodyType",
                     "fuelType", "gearbox", "power",
                     "kilometer", "notRepairedDamage", "regionCode",
                     "seller", "offerType", "creatDate",
                     "v_0", "v_1", "v_2",
                     "v_3", "v_4", "v_5",
                     "v_6", "v_7", "v_8",
                     "v_9", "v_10", "v_11",
                     "v_12", "v_13", "v_14"
                     ],
    "dataset_scale": {"name":1000000, "regDate":100000000,
                     "model":1000, "brand":100, "bodyType":10,
                     "fuelType":10, "gearbox":10, "power":100000,
                     "kilometer":100, "notRepairedDamage":10, "regionCode":10000,
                     "seller":10, "offerType":10, "creatDate":100000000,
                      "price":100000,
                     "v_0":100, "v_1":10, "v_2":100,
                     "v_3":10, "v_4":10, "v_5":1,
                     "v_6":1, "v_7":10, "v_8":1,
                     "v_9":1, "v_10":100, "v_11":100,
                     "v_12":100, "v_13":10, "v_14":10}

}

train_config = {
    "epoch": 15,
    "batch_size": 32,
    "learning_rate": 0.001,
    "model_ckpt_dir": './',
    "show_iteration":10,
    "model_load_path":None,
    "model_save_path":"./model_save_state/"
}
predict_config = {
    "model_predict_path":"./model_save_state/1_sale.pth",
    "batch_size": 1,
}
