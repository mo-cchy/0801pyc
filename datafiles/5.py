import pandas as pd
import os
from sklearn import tree
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

target_directory = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"

os.chdir(target_directory)


df = pd.read_csv('iris.csv')

# df['がく片長さ'] = df['がく片長さ'].fillna(df['がく片長さ'].mean())
# df['がく片幅'] = df['がく片幅'].fillna(df['がく片幅'].mean())
# df['花弁長さ'] =df['花弁長さ'].fillna(df['花弁長さ'].mean())
# df['花弁幅'] =df['花弁幅'].fillna(df['花弁幅'].mean())
# df['種類'] = df['種類'].fillna(df['種類'].mean())

xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']

# n_df = df.select_dtypes(include=['number'])
colmean = df[xcol].mean()
df2 = df.fillna(colmean)

x = df2[xcol]
t = df2['種類']

model = tree.DecisionTreeClassifier(max_depth=2,random_state=0)
# model.fit(x,t)

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.3,random_state=0)

model.fit(x_train,y_train)

# with open('irismodel.pkl','wb')as f:
#     pickle.dump(model,f)

print(model.tree_.feature)
print(model.tree_.threshold)

x_train.columus = ['gaku_nagasa','gaku_haba','kaben_nagasa','kaben_haba']
import matplotlib.pyplot as plt
plt.figure()
x = plot_tree(model,feature_names=x_train.columus,filled=True)
plt.show()