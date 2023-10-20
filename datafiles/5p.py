import pandas as pd
import os
from sklearn import tree
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

target_directory = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"

os.chdir(target_directory)

df = pd.read_csv('ex2.csv')

# print(df.head(3)) #練習5-2
# print(df['target'],df['target'].value_counts()) #練習5-4

# print(df.isnull(),df.isnull().any(axis=0)) #練習5-5

df['x1'] = df['x1'].fillna(df['x1'].median())
df['x2'] = df['x2'].fillna(df['x2'].median())

xcol = ['x0','x1','x2','x3']
x = df[xcol]
t = df['target']

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

model = tree.DecisionTreeClassifier(max_depth=3,random_state=0)

model.fit(x_train,y_train)
# print(model.score(x_train,y_train))
# print(model.score(x_test,y_test))

new = [[1.56,0.23,-1.1,-2.8]]
print(model.predict(new))