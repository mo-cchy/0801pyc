import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import os

target_directory = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"

os.chdir(target_directory)

# %matplotlib inline

df = pd.read_csv('Survived.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df2 = pd.read_csv('Survived.csv')

is_null = df2['Age'].isnull()

df2.loc[(df2['Pclass']==1)&(df['Survived']==0)&(is_null),'Age']=43
df2.loc[(df2['Pclass']==1)&(df['Survived']==1)&(is_null),'Age']=35

df2.loc[(df2['Pclass']==2)&(df['Survived']==0)&(is_null),'Age']=33
df2.loc[(df2['Pclass']==2)&(df['Survived']==1)&(is_null),'Age']=25

df2.loc[(df2['Pclass']==3)&(df['Survived']==0)&(is_null),'Age']=26
df2.loc[(df2['Pclass']==3)&(df['Survived']==1)&(is_null),'Age']=20

sex = df2.groupby('Sex')['Survived'].mean()

male = pd.get_dummies(df2['Sex'],drop_first=True)
E = pd.get_dummies(df2['Embarked'],drop_first=True)


col = ['Pclass','Age','SibSp','Parch','Fare','Sex']

x = df2[col]
t = df2['Survived']

x_temp = pd.concat([x,male],axis=1)

x_new = x_temp.drop('Sex',axis=1)

# x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

# model = tree.DecisionTreeClassifier(max_depth=5,random_state=0,class_weight='balanced')

# model.fit(x_train,y_train)

def learn(x,t,depth = 3):
    x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(x_train,y_train)

    score = model.score(X = x_train,y = y_train)
    score2 = model.score(X = x_test,y = y_test)
    return round(score,3),round(score2,3),model

# for j in range(1,6):
#     s1,s2,m = learn(x_new,t,depth=j)
#     s = '深さ{}:訓練データの精度{}::テストデータの精度{}'
#     print(s.format(j,s1,s2))

s1,s2,model = learn(x_new,t,depth=5)

import pickle
with open('survived.pkl','wb')as f:
    pickle.dump(model,f)