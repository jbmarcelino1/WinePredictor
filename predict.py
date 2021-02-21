import joblib
import pandas as pd
import os
from WinePredictModel.data import GetData
from termcolor import colored
feat = joblib.load("model/feature_eng.joblib")
model = joblib.load("model/model.joblib")

d = GetData('gcp')
df = d.clean_data()
test_x = df.iloc[1,:]
test_x = pd.DataFrame(test_x).T
test_x = test_x.drop(columns='points').reset_index(drop=True)
test_x.to_csv("/Users/edwardburroughes/Desktop/test.csv")
feat_eng_x = feat.transform(test_x)
ypred = model.predict(feat_eng_x)
print(colored(f"model prediction:{ypred}",'blue'))
