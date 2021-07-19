from m3inference import M3Twitter
from m3inference import M3Inference
import pprint
import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime
import multiprocessing as mp

# User's ID
def getUserID(dataset):
    item = dataset['user']['id']
    try:
        if item is not None:
            return str(item)
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# screen name
def getScreenName(dataset):
    item = dataset['user']['screen_name']
    try:
        if item is not None:
            return item.lower()
        else:
            raise TypeError
    
    except TypeError:
        print('None')

        
def run_sim(run):
    datasets = []
    vol_num = str(run + 1)
    m3_input_file = './us_presidential_election_2020/M3_vp_debate/twitter_cache_vol-'+vol_num+'/m3_input_'+'vol_'+vol_num+'.jsonl'
    cache_dir = './us_presidential_election_2020/M3_vp_debate/twitter_cache_'+'vol-'+vol_num
    df_save_file = './us_presidential_election_2020/M3_vp_debate/outputs/'+'vol-'+vol_num+'.csv'
    
    path = './us_presidential_election_2020/201006200213_vp_debate/'
    file_name = os.listdir(path)
    file_path = os.path.join(path, file_name[run])
    print("Processing:..."+file_path)
    for line in open(file_path, 'r'):
        datasets.append(json.loads(line))
    m3twitter=M3Twitter(cache_dir=cache_dir)
    m3twitter.transform_jsonl(input_file=file_path, output_file=m3_input_file)
    outputs = m3twitter.infer(m3_input_file)
    outputs = dict(outputs)
    
    uid_lst = []
    screen_name_lst = []
    age_lst = []
    gender_lst = []
    org_lst = []
    for i in range(len(outputs)):
        uid = getUserID(datasets[i])
        screen_name = getScreenName(datasets[i])
        age = outputs[uid]['age']
        gender = outputs[uid]['gender']
        org = outputs[uid]['org']

        uid_lst.append(uid)
        screen_name_lst.append(screen_name)
        age_lst.append(age)
        gender_lst.append(gender)
        org_lst.append(org)

    df = pd.DataFrame(uid_lst, columns=['UID'])
    df['screen_name'] = screen_name_lst
    df['age'] = age_lst
    df['gender'] = gender_lst
    df['org'] = org_lst

    df.to_csv(df_save_file, index=False)
    
def main():
    pool = mp.Pool(processes=100)
    for run in range(10, 78):
        pool.apply_async(run_sim, args=(run, ))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()