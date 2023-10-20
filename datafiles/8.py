import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

def learn(x,t):
    x_train,x_val,y_train,y_val = train_test_split(x,t,test_size=0.2,random_state=0)
    sc_model_x = StandardScaler() #インスタンスを作成
    sc_model_y = StandardScaler() #インスタンスを作成
    sc_model_x.fit(x_train) #標準化したデータの学習の実行
    sc_x_train = sc_model_x.transform(x_train) #標準化したデータを代入
    sc_model_y.fit(y_train) #標準化したデータの学習の実行
    sc_y_train = sc_model_y.transform(y_train) #標準化したデータを代入

    model = LinearRegression() #インスタンスを作成(線形回帰)
    model.fit(sc_x_train,sc_y_train) #学習の実行

    sc_x_val = sc_model_x.transform(x_val) #標準化したデータを代入
    sc_y_val = sc_model_y.transform(y_val) #標準化したデータを代入

    train_score = model.score(sc_x_train,sc_y_train)
    val_score = model.score(sc_x_val,sc_y_val)

    return train_score,val_score

TD = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"
os.chdir(TD) #絶対パスでディレクトリを移動

df = pd.read_csv('Boston.csv') #CSVファイルの読み込み

crime = pd.get_dummies(df['CRIME'],drop_first=True) #CRIME列をダミーに変更

df2 = pd.concat([df,crime],axis=1) #ダミー変数を連結
df2 = df2.drop(['CRIME'],axis=1) #元のCRIME列を削除

train_val,test = train_test_split(df2,test_size=0.2,random_state=0) #8:2で訓練データとテストデータに分割

train_val2 = train_val.fillna(train_val.mean()) #訓練データの欠損値に平均値で穴埋め

# colname = train_val2.columns
# for name in colname:
#     train_val2.plot(kind= 'scatter',x= name,y = 'PRICE') #各列とPRICE列を相関関係を示す
#     plt.show() #散布図を描く

# out_line1 = train_val2[(train_val2['RM']<6) & (train_val2['PRICE'])>40].index
# out_line2 = train_val2[(train_val2['PTRATIO']>18) & (train_val2['PRICE'])>40].index #外れ値がどこにあるか確認

# print(out_line1,out_line2)

train_val3 = train_val2.drop([76],axis=0) #外れ値を削除する

col = ['INDUS','NOX','RM','PTRATIO','LSTAT','PRICE'] #各要素を指定する
train_val4 = train_val3[col] #上記のデータを取り出し代入してあtらしいデータを作成する

train_cor = train_val4.corr()['PRICE'] #PRICE列を抜き出す

abs_cor = train_cor.map(abs) #絶対値に変換

col = ['RM','LSTAT','PTRATIO'] ##各要素を指定する

x = train_val4[col]
t = train_val4[['PRICE']] #後々に標準化させるのでシリーズではなくデータフレームで作る

x_train,x_val,y_train,y_val = train_test_split(x,t,test_size=0.2,random_state=0) #訓練データと検証データに分割

sc_model_x = StandardScaler() #インスタンスを作成
sc_model_x.fit(x_train) #標準化したデータの学習の実行

sc_x = sc_model_x.transform(x_train)

tmp_df = pd.DataFrame(sc_x,columns=x_train.columns)

sc_model_y = StandardScaler() #インスタンスを作成
sc_model_y.fit(y_train)

sc_y = sc_model_y.transform(y_train)

model = LinearRegression() #インスタンスを作成
model.fit(sc_x,sc_y)

sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)


x = train_val3.loc[:,['RM','LSTAT','PTRATIO']]
t = train_val3[['PRICE']]

x['RM2'] = x['RM'] ** 2
x['LSTAT2'] = x['LSTAT'] ** 2
x['PTRATIO2'] = x['PTRATIO'] ** 2

# x.loc[2000] = [10,7,8,100]
# print(x.tail(2))

x['RM * LSTAT'] = x['RM'] * x['LSTAT']

s1,s2 = learn(x,t)

sc_model_x2 = StandardScaler()
sc_model_y2 = StandardScaler()

sc_model_x2.fit(x)
sc_model_y2.fit(t)

sc_x = sc_model_x2.transform(x)
sc_y = sc_model_y2.transform(t)

model = LinearRegression()
model.fit(sc_x,sc_y)

test2 = test.fillna(train_val.mean())
x_test = test2.loc[:,['RM','LSTAT','PTRATIO']]
y_test = test2[['PRICE']]

x_test['RM2'] = x_test['RM'] ** 2
x_test['LSTAT2'] = x_test['LSTAT'] ** 2
x_test['PTRATIO2'] = x_test['PTRATIO'] ** 2

x_test['RM * LSTAT'] = x_test['RM'] * x_test['LSTAT']

sc_x_test = sc_model_x2.transform(x_test)
sc_y_test = sc_model_y2.transform(y_test)

with open('boston.pkl','wb') as f:
    pickle.dump(model,f)

with open('boston_scx.pkl','wb') as f:
    pickle.dump(sc_model_x2,f)

with open('boston_scy.pkl','wb') as f:
    pickle.dump(sc_model_y2,f)