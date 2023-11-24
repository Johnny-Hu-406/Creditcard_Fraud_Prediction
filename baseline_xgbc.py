import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pickle

# 載入訓練集
data = pd.read_csv("train_dropnameAndNUll.csv")

# Step 1 (原本設計z1是為了隨機挑選來欠採樣0樣本，後來xgboost訓練很快就沒有欠採樣了)
z1 = data[data['label'] == 0]
# Step 2
z2 = data[data['label'] == 1]

# Step 3
z3 = pd.concat([z1, z2])

# Step 4 (我覺得scity應該沒用就山地了)
X = z3.drop(['label', 'scity'], axis=1)
y = z3['label']

# Step 5 特徵選擇
selector = SelectKBest(f_classif, k='10')  # Change k value as needed
X_new = selector.fit_transform(X, y)
# 獲取選擇的特徵索引
selected_feature_indices = selector.get_support(indices=True)
# 獲取選擇的特徵名稱
selected_feature_names = X.columns[selected_feature_indices]

# Step 6 切割訓練測試集
# Assuming X_new and y are the selected features and labels after feature selection
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=3)

# Step 7 特徵正規化，方便訓練並降低不同訓練資料特徵維度的比例
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# precision很高
# max_depth= 20, min_child_weight= 7.734890981711013

# Step 8 設定XGBClassifier參數並訓練，供參考(好像大約1-score 45%)
model = XGBClassifier(max_depth= 20, 
                      min_child_weight= 7.734890981711013,
                      scale_pos_weight= 5, 
                      gamma = 3,random_state =42, 
                      eval_metric= "auc")
model.fit(X_train, y_train)

# Step 9: Model Evaluation
y_trainpred = model.predict(X_train)
y_pred = model.predict(X_test)

# Model evaluation
print("TrainingData Classification Report: ")
print(classification_report(y_train, y_trainpred))
print('*'*20)
print("y_pred Classification Report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

# Step 10: save Model 
pickle.dump(model, open("XGBC.pickle.dat","wb"))

# 載入測試集測試 這個測試集有經過preprocessing.py處理，把文字特徵刪除
public_processed_df = pd.read_csv("public_processed_dropname.csv")
public_feature = public_processed_df.drop(['txkey', 'scity'], axis=1)
public_name = public_processed_df['txkey']

# 自己設計的資料清理
public_feature["stscd"].fillna(1, inplace=True)
public_feature["stocn"].fillna(0, inplace=True)
mean_values = public_feature.mean()
public_no_null = public_feature.fillna(mean_values)
public_training_5 = public_no_null[selected_feature_names]

# 使用訓練集的sc進行縮放
public_X_new = sc.transform(public_training_5)

# 使用上面的xgbc測試集辨識
public_y_pred = model.predict(public_X_new)

# 製作與儲存可以上傳的csv檔案
public_y_pred = pd.DataFrame(public_y_pred, columns=["pred"])
result = pd.concat([public_name, public_y_pred], axis=1)
result.to_csv('result.csv',index=False)