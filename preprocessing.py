# 將訓練集刪除無用的字元特徵
import pandas as pd

# 載入訓練集
train_data = pd.read_csv("dataset_1st\public_processed.csv")

# 列出要刪除的欄位
# drop_cols = ["txkey", "chid", "cano", "mchno", "acqic"]
drop_cols = ["chid", "cano", "mchno", "acqic"]

# 刪除指定的欄位
train_data = train_data.drop(columns=drop_cols)

# 儲存刪除後的訓練集
train_data.to_csv("public_processed_dropname.csv", index=False)
