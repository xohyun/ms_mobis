import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import sklearn

# 전처리한 csv 불러오기
def data_loading(path, subject_num, day, expt) :
  preprocessed_data=pd.read_csv(path + "S%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))
  preprocessed_data=preprocessed_data.drop(["Unnamed: 0"],axis=1) # preprocess 다시 돌려서 나온 데이터 사용시 이부분 제거
  eeg_time_data = preprocessed_data.drop(["Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1)
  eeg_data = preprocessed_data.drop(["Time (s)", "Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1) # time이랑 packet counter column 제거
  eeg_col_names = list(preprocessed_data.columns)  # list로
  return preprocessed_data, eeg_time_data, eeg_data, eeg_col_names # name에는 time이 포함되어 있음

# array형식으로 만들기
def create_array(dfdata, col_names, fs, ch_types, scale = 1) : 
  info = mne.create_info(ch_names = col_names, sfreq = fs, ch_types = ch_types)
  temp = dfdata.values
  temp = temp.T * scale
  s_array = mne.io.RawArray(temp, info)
  return s_array

# bandpass filtering 수행하기
def bandpass_filter(arrdata, preprocessed_data) :
  filtered_data = arrdata.filter(l_freq = 1., h_freq=50.,  picks='eeg', fir_design='firwin', skip_by_annotation='edge')
  
  filtered_df = filtered_data.to_data_frame() # filtered_df : filtering한 후, 전체 변수 다 있으면서 데이터프레임 형식. (time include)
  filtered_df['Time (s)'] = preprocessed_data['Time (s)']
  filtered_df = filtered_df.drop(['time'], axis = 1) # 자동으로 생기는 time변수 제거
  
  filtered_df_trigger = filtered_df.copy()
  filtered_df_trigger['TRIGGER(DIGITAL)'] = preprocessed_data['TRIGGER(DIGITAL)']
  return filtered_data, filtered_df, filtered_df_trigger
  
# ICA
def ica_processing(f_arr) :
  ica = mne.preprocessing.ICA(n_components = 28, random_state = 97, max_iter = 800)
  ica.fit(f_arr)
  ica.exclude = [0,1] # 몇 번 요소 선택할지 결정
  ica.plot_properties(f_arr, picks = ica.exclude)
  ica.fit(f_arr)

  f_arr.load_data()
  ica.plot_sources(f_arr, show_scrollbars = True)
  #plt.show()

# plot 보여주는 함수
def show_plot(arrdata) :
  arrdata.plot()
  plt.show()

# trigger 시간순으로 보여주는 plot
def trigger_plot(preprocessed_data) :
  plt.figure(figsize = (15,5))
  plt.plot(preprocessed_data.iloc[:,0:1], preprocessed_data.iloc[:,34:35], color = "blue")
  plt.xlabel('Time (sec)')
  plt.ylabel('Sickness level')
  plt.show()

############################################################################
################################### main ###################################
############################################################################

# 설정
data_path = "C:\\Users\\soso\\Desktop\\store\\" # 데이터 경로 설정
subject_num = 13 # 피험자 번호
day = 1  # 1/2 
expt = 2 # 1/2 
preprocessed_data, eeg_time_data, eeg_data, eeg_col_names = data_loading(data_path, subject_num, day, expt)
fs = 500 # freq 500hZ 설정
ch_names = ["Fp1","Fp2","AF3","AF4","F7","F8","F3","Fz","F4","FC5",
            "FC6","T7","T8","C3","C4","CP5","CP6","P7","P8","P3","Pz",
            "P4","PO7","PO8","PO3","PO4","O1","O2"] # 채널 이름 설정
ch_types = ["eeg"] * 28
montage = mne.channels.make_standard_montage("standard_1005")

s_array = create_array(eeg_data, ch_names, fs, ch_types, scale = (1e-6))
#show_plot(s_array)

montage = mne.channels.make_standard_montage("standard_1005")
s_array.set_montage(montage = montage)

filtered_arr, filtered_df, filter_df_trigger = bandpass_filter(s_array, preprocessed_data)
#show_plot(filtered_arr)
filtered_eeg = filtered_df.iloc[:,0:28]
s_array_filter = create_array(filtered_eeg, ch_names, fs, ch_types, scale = (1e-6))
s_array_filter.set_montage(montage = montage)
# s_array_filter.plot_psd()
s_array_filter.plot(duration = 5, n_channels = 28)
plt.show()

# ica_processing(filtered_arr)
events = mne.make_fixed_length_events(s_array_filter, duration = 5)
epochs = mne.Epochs(s_array_filter, events)

# trigger_plot(preprocessed_data) #trigger 상황 파악

##############################################################################
##############################################################################
# 데이터 세 분류로 나누기 --> 분류 어떻게 할지는 고민해봐야..(지금 이건 주관적 판단)
baseline = preprocessed_data.loc[preprocessed_data['Time (s)'] < 201].copy()
# low_level = eeg_time_data.loc[(eeg_time_data['Time (s)'] >= 300) & (eeg_time_data['Time (s)'] < 501)].copy()
# high_level = eeg_time_data.loc[(eeg_time_data['Time (s)'] >= 700) & (eeg_time_data['Time (s)'] < 901)].copy()
baseline_eeg_time_data = baseline.drop(["Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1)
baseline_eeg_data = baseline.drop(["Time (s)", "Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1) # time이랑 packet counter column 제거
baseline_eeg_col_names = list(baseline.columns)  # list로

baseline_s_array = create_array(baseline_eeg_data, ch_names, fs, ch_types, scale = (1e-6))
baseline_s_array.set_montage(montage = montage)


# filtered_arr.plot()
# print(eeg_time_data)
# print(baseline)

baseline_farr, baseline_fdf, baseline_fdf_trigger = bandpass_filter(baseline_s_array, baseline)
baseline_filtered_eeg = baseline_fdf.iloc[:,0:28]
baseline_farr.set_montage(montage = montage)
baseline_farr.plot(duration = 5, n_channels = 28)
#s_array_filter.plot_psd()

# #filtered_arr.plot(duration = 5, n_channels = 28)
baseline_farr.plot(duration = 5, n_channels = 28)


baseline_farr.plot_psd(average = True)
baseline_farr.crop(tmax = 5).load_data()
baseline_farr.plot_psd_topo()
plt.show()