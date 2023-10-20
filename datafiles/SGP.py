import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# sklearn.model_selection: データセットの分割や交差検証のためのモジュール
# sklearn.preprocessing: データの前処理（標準化、ラベルエンコーディングなど）のためのモジュール
# sklearn.tree: 決定木モデルのためのモジュール
# sklearn.metrics: モデルの評価指標（精度、混同行列など）を計算するためのモジュール

def learn(x,t,depth = 10):
    x_train,x_val,y_train,y_val = train_test_split(x,t,test_size=0.2,random_state=0)

    model = tree.DecisionTreeClassifier(max_depth=depth,random_state=0,class_weight='balanced')
    model.fit(x_train,y_train)

    score = model.score(X = x_train,y = y_train)
    score2 = model.score(X = x_val,y = y_val)
    return round(score,3),round(score2,3),model


TD = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"
os.chdir(TD)

df = pd.read_csv('student_graduation_prediction.csv')

df['CampusActivity'] = df['CampusActivity'].fillna('None')
df = df.drop(['GPA'],axis=1)

str_col_name = ['SocialActivity','PartTimeExperience','Scholarship','Major','TuitionPayment','LivingSituation','Hobbies','CampusActivity','GraduationPrediction']
str_df = df[str_col_name]
str_df = pd.get_dummies(df[str_col_name],drop_first=True)

df2 = pd.concat([df,str_df],axis=1)
df2 = df2.drop(str_col_name,axis=1)

train_val,test = train_test_split(df2,test_size=0.2,random_state=0)

# col = ['AttendanceDays','YearsEnrolled','SocialActivity_Yes','Scholarship_Yes']
x = train_val.drop(['GraduationPrediction_Graduate'],axis=1)
# x= train_val[col]
t = train_val['GraduationPrediction_Graduate']

graduationPrediction = df2.groupby('GraduationPrediction_Graduate').mean()
graduationPrediction['AttendanceDays'].plot(kind='bar')
plt.show()

# x_train,x_val,y_train,y_val = train_test_split(x,t,test_size=0.2,random_state=0)

# model = tree.DecisionTreeClassifier(max_depth=10,random_state=0,class_weight='balanced')
# model.fit(x_train,y_train)

# # print(df2.isnull().sum())

# print(model.score(x_val,y_val))
