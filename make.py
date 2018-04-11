#!/usr/bin/env python
import os
from tqdm import tqdm
from glob import glob
import pandas as pd

from extraction import main as extract
from analysis import (
    estimate_darkcount_rate_max_min,
    estimate_darkcount_rate_random_charge,
)


out_dir = 'extraction_output'
os.makedirs(out_dir, exist_ok=True)
input_dir = 'observations'

for path in tqdm(glob(input_dir+'/*')):

    # my own convention:
    # digicamtoy MCs don't have a field named digicam_baseline
    # So here we have to estimate the baseline ourselves.
    if 'toy' in path:
        use_digicam_baseline = False
    else:
        use_digicam_baseline = True

    outpath = path.replace(input_dir, out_dir)+'.h5'
    if not os.path.isfile(outpath):
        extract(
            input_file=path,
            output_file=outpath,
            use_digicam_baseline=use_digicam_baseline,
        )

dfs = []
for path in glob(out_dir+'/*'):

    df = estimate_darkcount_rate_max_min(path)
    df['method'] = 'max_min'
    df['path'] = path
    dfs.append(df)

    df = estimate_darkcount_rate_random_charge(path)
    df['method'] = 'random_charge'
    df['path'] = path
    dfs.append(df)

df = pd.concat(dfs)

df.to_hdf('results.h5', key='all')
