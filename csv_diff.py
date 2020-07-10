import pandas as pd
import numpy as np
df_submit305 = pd.read_csv(r'D:\360极速浏览器下载\submit_v30 (5).csv') 
df_submit291 = pd.read_csv(r'D:\360极速浏览器下载\submit_v29 (1).csv') 
# df_submit22 = pd.read_csv(r'D:\360极速浏览器下载\submit_v22.csv')      
# df_submit23 = pd.read_csv(r'D:\360极速浏览器下载\submit_v23.csv')      
a = [np.array(df_submit305['file_code'])== np.array(df_submit291['file_code'])]
np.savetxt("diff_305-291.txt", np.array(a[0]))