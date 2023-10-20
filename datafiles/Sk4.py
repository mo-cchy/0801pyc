import pickle
import os

target_directory = "c:/Users/0801_PyC/Desktop/sukkiri-ml-codes/datafiles/"

os.chdir(target_directory)

with open('KinokoTakenoko.pkl','rb')as f:
    model2 = pickle.load(f)

suzuki=[[180,75,30]]

print(model2.predict(suzuki))