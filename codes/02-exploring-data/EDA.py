import sys
sys.path.append('C:/Users/Eric/')

import pandas as pd
import seaborn_visualizer as sbv
import seaborn as sns

df = pd.read_csv('../../data/01-modified-data/big_five_final.csv')
sbv.get_pd_info(df)
sbv.pd_general_plots(df)