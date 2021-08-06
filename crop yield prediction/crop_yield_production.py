# -*- coding: utf-8 -*-
"""Crop yield production.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LkkBUnHaft-uTjxRKhOy9_keGq6YkpQb
"""

import numpy as np
import pandas as pd

# Visualisation Libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# from catboost import CatBoostRegressor


data=pd.read_csv('out.csv')
data.isna().sum()
#to check if any missing is present

#print(data['Production'][data['State']=='Andaman and Nicobar Islands'])

#Remove the index variable


data = data.drop('Unnamed: 0', axis=1)

df3 = data[data['Production'] < 50000]
data = df3[df3['Production']  >10 ]

#To check relation (positive/negative) between the Area, Production & Rainfall
APR_df = data[['Area','Production','Rainfall']]

corr = APR_df.corr()
plt.figure(figsize=(8, 8))
g = sns.heatmap(corr, annot=True, cmap = 'PuBuGn_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})
g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
g.set_title("Correlation between Area, Production & Rainfall", fontsize=14)
plt.show()

data.groupby('Crop').count()

data1 = data.drop(['Year'], axis=1)
data.head()

from sklearn.preprocessing import LabelEncoder
data1=data1[['Season','State','Crop']].apply(LabelEncoder().fit_transform)
data[['Season','State','Crop']]=data1
data.head()

data=data.drop(['Year'], axis=1)
label=data['Production']
traindata=data.drop(['Production'], axis=1)

traindata.head()
print(traindata.head())

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(traindata, label, test_size=0.3, random_state=42)

from sklearn.metrics import r2_score


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn import svm
# from sklearn.tree import DecisionTreeRegressor

# models = [
#     GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0),
#      RandomForestRegressor(n_estimators=200, max_depth=3, random_state=0),
#     svm.SVR(),
#    DecisionTreeRegressor()
# ]

# model_train=list(map(compare_models,models))

# print(*model_train, sep = "\n")

from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=0)
fit=model.fit(train_data,train_labels)

pickle.dump(model, open('model1.pkl','wb'))

y_pred=fit.predict(test_data)
r2=r2_score(test_labels,y_pred)
print(model.predict([[0,1,0,1254,2763]]))