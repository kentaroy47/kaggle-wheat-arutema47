# get a df with just image and source columns
# such that dropping duplicates will only keep unique image_ids
from sklearn.model_selection import StratifiedKFold
from itertools import islice
import pandas as pd
import numpy as np
import re

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

def make_dfsplits(config):
    DIR_INPUT = config.data["name"]
    DIR_TRAIN = f'{DIR_INPUT}/train'
    DIR_TEST = f'{DIR_INPUT}/test'

    train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)
    
    df = pd.read_csv(f'{DIR_INPUT}/train.csv')
    image_source = df[['image_id', 'source']].drop_duplicates()

    # get lists for image_ids and sources
    image_ids = image_source['image_id'].to_numpy()
    sources = image_source['source'].to_numpy()

    # do the split
    # in other words:
    # split up our data into 10 buckets making sure that each bucket
    #  has a more or less even distribution of sources
    # Note the use of random_state=1 to ensure the split is the same each time we run this code
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    split = skf.split(image_ids, sources) # second arguement is what we are stratifying by

    # we can use islice to control which split we select
    select = 0
    train_ix, val_ix = next(islice(split, select, select+1))

    # translate indices to ids
    train_id = image_ids[train_ix]
    val_id = image_ids[val_ix]

    # create corresponding dfs
    val_df = train_df[train_df['image_id'].isin(val_id)]
    train_df = train_df[train_df['image_id'].isin(train_id)]

    return train_id, val_id, train_df, val_df