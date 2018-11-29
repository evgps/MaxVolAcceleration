import os
import numpy as np
import pickle as pkl
import matplotlib.pylab as plt
import pandas as pd
from feedforwardcompressed import slice_blocks, get_batch
from os import listdir
from os.path import isfile, join

import torch
from collections import defaultdict
import dataloaders


def format_time(x):
    x = (x if not pd.notnull(x) else x.average)
    return x

def get_rate(x):
    return x[-1]

def get_notcompressed_count(x):
    return len(np.where(x==1)[0])

def get_modellabel(x):
    x, devise = x.rsplit('_',1)
    rs = np.array([1 if el.strip()=='None' else float(el.strip()) for
                   el in x.split('[')[1].strip('].pth').split(',')])
    
    compressed_count = (sum(rs!=1) if len(rs)!=sum(rs) else 0)
    return "{}x, {}".format(int(1/rs[-1]), compressed_count)


def get_total_df(dataframes, total_dataframes, dataset = None, arch=None, device = None):
    if device == 'cuda':
        device = 1
    else:
        device = 0

    print(device)
    df = dataframes[dataset][device].copy()
    
    print(df.shape)
    if dataset.startswith('cifar'):
        df = df[list(filter(lambda col: col.startswith(arch.split('.')[0]), df.columns))]
            
    
    for column_name in df.columns:
        df[column_name] = df[column_name].apply(lambda x: format_time(x) if(np.all(pd.notnull(x))) else x)

    rates = [np.array([1. if el.strip()=='None' else float(el.strip()) for
                       el in column_name.split('].')[0].split('_[')[-1].split(',')])
             for column_name in df.columns]

    df.fillna(value=0., inplace = True)
    # total_time = df.sum()

    df = df.T
    # df['conv_time'] = total_time

    df['rates'] = rates
    df['rate'] = df['rates'].apply(get_rate)
    df['notcompressed_count'] = df['rates'].apply(get_notcompressed_count)

    if dataset == 'stl10':
        add = 0
    else:
        add = 1
    df['compressed_time'] = df.apply(lambda row: sum(row[row['notcompressed_count']:len(row['rates'])+add]), axis=1)
    df['notcompressed_time'] =  df.apply(lambda row: sum(row[:row['notcompressed_count']]), axis=1) 
    df['conv_time'] = df['notcompressed_time']+df['compressed_time']
    

    inference_df = total_dataframes[dataset][device].copy()
    inference_df = inference_df[list(filter(lambda col: col.startswith('vgg'), inference_df.columns))]

    for column_name in inference_df.columns:
        inference_df[column_name] = inference_df[column_name].apply(lambda x: format_time(x) if(np.all(pd.notnull(x))) else x)
    inference_df = inference_df.T
    inference_df = inference_df.rename(index=str, columns={0: "inference_time"})


    total_df = pd.concat([df, inference_df], sort = False, axis = 1)
    total_df.index = pd.Index([get_modellabel(x) for x in total_df.index])
    
    return total_df