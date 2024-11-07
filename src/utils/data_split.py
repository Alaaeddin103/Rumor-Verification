import pandas as pd
import json
from sklearn.model_selection import train_test_split

#load and combine train and dev data from the shared task 
def load_and_combine_datasets(train_file, dev_file):
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    combined_data = train_data + dev_data
    return combined_data

#perform stratified shuffle split
def stratified_split(data, label_key, test_size=0.2, val = False):
    labels = [item[label_key] for item in data]
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=labels, random_state=42)
    if val:
        labels = [item[label_key] for item in train_data]
        train_data, val_data = train_test_split(train_data, test_size=test_size, stratify=labels, random_state=42)
        return train_data, test_data, val_data
    else:
        return train_data, test_data