import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def learn(x,t,depth = 10):
    x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(x_train,y_train)

    score = model.score(X = x_train,y = y_train)
    score2 = model.score(X = x_test,y = y_test)
    return round(score,3),round(score2,3),model


TD = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"
os.chdir(TD)

df = pd.read_csv('Bank.csv')

df['duration'] = df['duration'].fillna(df['duration'].mean())
# print(df.isnull().sum())

# print(df['job'].value_counts())
# print(df['marital'].value_counts())
# print(df['education'].value_counts())
# print(df['default'].value_counts())
# print(df['loan'].value_counts())
# print(df['contact'].value_counts())
# print(df['month'].value_counts())

df['housing'] = pd.get_dummies(df['housing'],drop_first=True)
df['default'] = pd.get_dummies(df['default'],drop_first=True)
df['loan'] = pd.get_dummies(df['loan'],drop_first=True)

job = pd.get_dummies(df['job'],drop_first=True)
marital = pd.get_dummies(df['marital'],drop_first=True)
education = pd.get_dummies(df['education'],drop_first=True)
contact = pd.get_dummies(df['contact'],drop_first=True)
month = pd.get_dummies(df['month'],drop_first=True)


df2 = pd.concat([df,marital,education,contact,month],axis=1)
df2 = df2.drop(['marital','education','contact','day','month'],axis=1)

df2 = pd.concat([df2,job],axis=1)
df2 = df2.drop(['job'],axis=1)


# colname = train_val.columns
# for name in colname:
#     train_val.plot(kind = '',x=name,y='y')
#     plt.show()

# train_val,test = train_test_split(df2,test_size=0.2,random_state=0)

# col = ['age','default','amount','housing','loan','duration','previous','campaign','married','single','secondary','tertiary','unknown','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed']
# x = train_val[col]

x = df2.drop(['id','y'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,t,test_size=0.2,random_state=0)

model = tree.DecisionTreeClassifier(max_depth=10,random_state=0,class_weight='balanced')
model.fit(x_train,y_train)



# threshold = 0.01

# feature_importances = model.feature_importances_
# important_features = [feature for feature, importance in enumerate(feature_importances) if importance > threshold]

# for feature_index in important_features:
#     print(f"Feature {feature_index}: Importance {feature_importances[feature_index]}")

for j in range(1,10):
    trS,teS,model = learn(x,t,depth=j)
    S1 = '訓練データ{}'
    S2 = 'テストデータ{}'
    T = '深さ{}:'+S1+S2
    print(T.format(j,trS,teS))


print(model.score(x_test,y_test))



