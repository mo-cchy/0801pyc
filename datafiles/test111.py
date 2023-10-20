import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

TD = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"
os.chdir(TD)

df = pd.read_csv('test.csv')

print(df.isnull().any(axis=0))
