import pandas as pd 
import numpy as np 
import os

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

train_processed_data = train_data.fillna(train_data.mean()) 
test_processed_data = train_data.fillna(test_data.mean()) 

data_path = os.path.join('data','processed')

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False) 
