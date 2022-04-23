import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import sklearn

data_path_drive = 'E:\\data\\PREPROCESSED_DATA'
data_path_rest = 'E:\\data\\rest'

day = 1
expt = 1

for i in range(1, 24) :
    subject_num = i
    if subject_num < 10 :
        preprocessed_drive = pd.read_csv(data_path_drive + "\\S0%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))
        preprocessed_rest = pd.read_csv(data_path_rest + "\\S0%d_DAY%d_EXPT%d_REST_preprocess.csv" %(subject_num, day, expt))
    else :
        preprocessed_drive = pd.read_csv(data_path_drive + "\\S%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))
        preprocessed_rest = pd.read_csv(data_path_rest + "\\S%d_DAY%d_EXPT%d_REST_preprocess.csv" %(subject_num, day, expt))

    concating_drive = preprocessed_drive.copy()
    addtime = preprocessed_rest['Time (s)'][len(preprocessed_rest)-1] # rest의 마지막 시간
    concating_drive_time = preprocessed_drive['Time (s)'] + addtime
    concating_drive['Time (s)'] = concating_drive_time.to_frame()

    result = pd.concat([preprocessed_rest, concating_drive])

    result.to_csv('C:\\Users\\soso\\Desktop\\blah\\S%d_DAY%d_EXPT%d_concatdata.csv'%(subject_num, day, expt), header = True, index = False)