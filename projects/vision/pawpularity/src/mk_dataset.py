import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data, num_splits):
    data['kfold'] = -1

    num_bins = int(np.floor(1 + np.log2(len(data))))

    data.loc[:,'bins'] = pd.cut(data['Pawpularity'], bins=num_bins, labels=False)

    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

    for f,(t_, v_) in enumerate(kf.split(X=data, y= data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop('bins', axis=1)

    return data


if __name__ == '__main__':
    df = pd.read_csv("./data/train.csv")
    df_5 = create_folds(df, num_splits=5)
    df_10 = create_folds(df, num_splits=10)
    
    df_5.to_csv('./data/train_5folds.csv',index=False)
    df_10.to_csv('./data/train_10folds.csv',index=False)