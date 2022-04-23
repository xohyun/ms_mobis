# 보고서 ecg
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import sklearn

# 전처리한 csv 불러오기
def data_making(preprocessed_data) :
    #preprocessed_data=preprocessed_data.drop(["Unnamed: 0"],axis=1) # preprocess 다시 돌려서 나온 데이터 사용시 이부분 제거
    #eeg_time_data = preprocessed_data.drop(["Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1)
    #eeg_data = preprocessed_data.drop(["Time (s)", "Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1) # time이랑 packet counter column 제거
    eeg_col_names = list(preprocessed_data.columns)  # list로
    r_data = preprocessed_data[['Time (s)','ECG.(uV)','TRIGGER(DIGITAL)']]
    return r_data # name에는 time이 포함되어 있음

def time_plot(preprocessed_data) :
    # trigger 시간순으로 보여주는 plot
    plt.figure(figsize = (15,5))
    plt.plot(preprocessed_data.iloc[:,0:1], preprocessed_data.iloc[:,1:2], color = "blue")
    #plt.plot(preprocessed_data.iloc[:,1:2], preprocessed_data.iloc[:,0:1], color = "blue")
    plt.xlabel('Time (ms)')
    plt.ylabel('ECG.(uV)')
    #plt.ylim([0, 2600])
    plt.show()

def array_make(dfdata) :
    fs = 500
    ch_types = ["ecg"]
    info = mne.create_info(ch_names = list(dfdata.columns), sfreq = fs, ch_types = ch_types)
    temp = dfdata.values
    temp = temp.T
    s_array = mne.io.RawArray(temp, info)
    return s_array

def bandpass_filter(arrdata, preprocessed_data) :
  filtered_data = arrdata.filter(l_freq = 1., h_freq=50.,  picks='ecg', fir_design='firwin', skip_by_annotation='edge')
  
  filtered_df = filtered_data.to_data_frame() # filtered_df : filtering한 후, 전체 변수 다 있으면서 데이터프레임 형식. (time include)
  filtered_df['Time (s)'] = preprocessed_data['Time (s)'].copy()
  #filtered_df = filtered_df.drop(['time'], axis = 1) # 자동으로 생기는 time변수 제거1
  
  filtered_df_trigger = filtered_df.copy()
  filtered_df_trigger['TRIGGER(DIGITAL)'] = preprocessed_data['TRIGGER(DIGITAL)']
  return filtered_data, filtered_df #, filtered_df_trigger

####################################################################################
#                                     main                                         #
####################################################################################
data_path = "C:\\Users\\soso\\Desktop\\data\\PREPROCESSED_DATA" # 데이터 경로 설정
day = 1  # 1/2 
expt = 1 # 1/2 

choose = [9, 10, 13, 20] #출력할 피험자 번호. 전체 출력하려면 range(1,24)
store_results = pd.DataFrame(index = range(0,23), columns = range(0, 21)) #전체 피험자의 시간별 ecg 저장하는 데이터프레임

for i in choose : #전체는 23명    
    one_subject = []        # ecg 잘라서 저장
    one_subject_ecg = []    # ecg만 저장
    one_subject_arr = []    # array 타입
    print("====================================%d===================================="%i)
        
    subject_num = i # 피험자 번호
    if (i >= 10) :
        preprocessed_data=pd.read_csv(data_path + "\\S%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))
    else :
        preprocessed_data=pd.read_csv(data_path + "\\S0%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))
    
    r_data = data_making(preprocessed_data)
    
    for j in range(0, 21) :
        one_subject.append(r_data.loc[(30000*j):(30000*(j+1))].copy())
        one_subject_ecg.append(one_subject[j]['ECG.(uV)']); one_subject_arr.append(array_make(one_subject_ecg[j].to_frame()))
        one_person_filter_arr, one_person_filter_df = bandpass_filter(one_subject_arr[j], one_subject[j])
        a = mne.preprocessing.find_ecg_events(one_person_filter_arr, ch_name = 'ECG.(uV)')
        store_results.loc[i-1, j] = a[2]

    # 확인용 그림
    # time_plot(one_subject[1])

print(store_results)
# 데이터프레임을 excel로 저장
#store_results.to_excel("C:\\Users\\soso\\Desktop\\data\\ecg_averagepulse_expt1_4(min).xlsx", index = False)