import pandas as pd
import os
from sklearn import tree
import pickle

target_directory = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"

os.chdir(target_directory)

# for filename in os.listdir():
#     print(filename)

df = pd.read_csv('KvsT.csv')

xcol = ['身長','体重','年代']
x = df[xcol]

t = df['派閥']


model = tree.DecisionTreeClassifier(random_state=0)

model.fit(x,t)

taro = [[170,70,20]]
matsuda = [172,65,20]
asagi = [158,48,20]

new_data = taro+[matsuda,asagi]

model.predict(new_data)

a=model.score(x,t)

with open('KinokoTakenoko.pkl','wb') as f:
    pickle.dump(model,f)