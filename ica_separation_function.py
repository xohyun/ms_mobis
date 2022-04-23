import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import sklearn

# 전처리한 csv 불러오기
def data_making(preprocessed_data) :
  #preprocessed_data=preprocessed_data.drop(["Unnamed: 0"],axis=1) # preprocess 다시 돌려서 나온 데이터 사용시 이부분 제거
  eeg_time_data = preprocessed_data.drop(["Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1)
  eeg_data = preprocessed_data.drop(["Time (s)", "Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","TRIGGER(DIGITAL)"], axis=1) # time이랑 packet counter column 제거
  eeg_col_names = list(preprocessed_data.columns)  # list로
  return eeg_time_data, eeg_data, eeg_col_names # name에는 time이 포함되어 있음

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
  plt.show()
  ica.plot_components()

  reconst_raw = f_arr.copy()
  ica.apply(reconst_raw)

  #artifact_picks = mne.pick_channels_regexp(f_arr.ch_names) #, regexp = regexp 
  artifact_picks = mne.pick_channels(f_arr.ch_names, include = f_arr.ch_names)
  reconst_raw.plot(order = artifact_picks, n_channels = len(artifact_picks), show_scrollbars = True)
  plt.show()

  ##추가부분
  montage = mne.channels.make_standard_montage("standard_1005")
  reconst_raw.set_montage(montage = montage)
  reconst_raw.plot_psd(fmin = 3.75, fmax = 30.25) ##
  reconst_raw.plot(duration = 5, n_channels = 28)
  plt.show() ##
  

def show_plot(arrdata) :
    # plot 보여주는 함수
    arrdata.plot()
    plt.show()

def trigger_plot(preprocessed_data) :
    # trigger 시간순으로 보여주는 plot
    plt.figure(figsize = (15,5))
    plt.plot(preprocessed_data.iloc[:,0:1], preprocessed_data.iloc[:,34:35], color = "blue")
    plt.xlabel('Time (sec)')
    plt.ylabel('Sickness level')
    plt.show()

def ica_separate_processing(preprocessed_data) :
    # filtering 및 ica, plot_psd, psd_topo 수행
    # 원하는 그림이 있다면 주석을 풀고 실행하면 됨
    eeg_time_data, eeg_data, eeg_col_names = data_making(preprocessed_data)
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
    show_plot(filtered_arr)
    filtered_eeg = filtered_df.iloc[:,0:28]
    s_array_filter = create_array(filtered_eeg, ch_names, fs, ch_types, scale = (1e-6))
    s_array_filter.set_montage(montage = montage)
    s_array_filter.plot_psd() ##
    s_array_filter.plot(duration = 5, n_channels = 28)
    plt.show() ##

    ica_processing(filtered_arr)
    events = mne.make_fixed_length_events(s_array_filter, duration = 5)
    epochs = mne.Epochs(s_array_filter, events)
 
    # filtered_arr.plot(duration = 5, n_channels = 28)
    #trigger_plot(preprocessed_data) #trigger 상황 파악

    # plot_psd_topo 그리기
    # s_array_filter.plot_psd(average = True)
    # s_array_filter.crop(tmax = 5).load_data()
    # s_array_filter.plot_psd_topo()
    # plt.show()
    
############################################################################
################################### main ###################################
############################################################################

# 설정
data_path = "C:\\Users\\soso\\Desktop\\data\\PREPROCESSED_DATA" # 데이터 경로 설정
subject_num = 13 # 피험자 번호
day = 1  # 1/2 
expt = 2 # 1/2 
preprocessed_data=pd.read_csv(data_path + "\\S%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" %(subject_num, day, expt))

# 전체 데이터 호출
ica_separate_processing(preprocessed_data)

# 데이터 분할 
#baseline = preprocessed_data.loc[preprocessed_data['Time (s)'] < 201].copy()
#low_level = preprocessed_data.loc[(preprocessed_data['Time (s)'] >= 300) & (preprocessed_data['Time (s)'] < 501)].copy()
#high_level = preprocessed_data.loc[(preprocessed_data['Time (s)'] >= 700) & (preprocessed_data['Time (s)'] < 901)].copy()

# 데이터 호출
#ica_separate_processing(baseline)
#ica_separate_processing(low_level)
#ica_separate_processing(high_level)
