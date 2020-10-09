#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:18:36 2020

@author: FelipeAffonso
"""

import pandas as pd



gs = pd.read_csv('gender_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


trainidx =  train.shape[0]
testidx = test.shape[0]

passagerId = test['PassengerId']

target = train.Survived.copy()
train.drop(['Survived'], axis = 1, inplace=True)


df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

df_merged.shape
df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace=True)

df_merged.isnull().sum()

df_merged['Age'].fillna(df_merged['Age'].median(), inplace=True)
df_merged['Fare'].fillna(df_merged['Fare'].median(), inplace=True)
df_merged['Embarked'].fillna(df_merged['Embarked'].value_counts()[0], inplace=True)


df_merged['Sex'] = df_merged['Sex'].map({'male':0, 'female':1})

embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop(['Embarked'], axis = 1, inplace=True)

df_merged.head()

#scaling variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_merged = scaler.fit_transform(df_merged)





train = df_merged.iloc[:trainidx]
test = df_merged.iloc[trainidx:]
train.shape
test.shape


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model = LogisticRegression()
model.fit(train, target)
acc_reg_log = round(model.score(train,target)*100,2)
print(acc_reg_log)


model = DecisionTreeClassifier()
model.fit(train,target)
acc_tree = round(model.score(train,target)*100,2)
print(acc_tree)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train,target)
acc_tree = round(model.score(train,target)*100,2)
print(acc_tree)



y_pred = model.predict(test)
sub = pd.DataFrame({'PassengerId':passagerId, 'Survived':y_pred})

sub.to_csv('Submission_Titanic.csv', index=False)