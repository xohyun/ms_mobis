import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import sklearn

data_path = "C:\\Users\\soso\\Desktop\\store\\"
preprocessed_data=pd.read_csv(data_path + "S13_DAY1_EXPT2_DRIVE_preprocess.csv",)
preprocessed_data=preprocessed_data.drop(["Unnamed: 0"],axis=1) # preprocess 다시 돌려서 나온 데이터 사용시 이부분 제거
eeg_data = preprocessed_data.drop(["Time (s)", "Packet Counter(DIGITAL)","ECG.(uV)","Resp.(Ω)",   "PPG(ADU)",   "GSR(Ω)","Packet Counter(DIGITAL)","TRIGGER(DIGITAL)"], axis=1) # time이랑 packet counter column 제거
print(eeg_data.head())
fs = 250 # freq 250hZ

col_names = list(preprocessed_data.columns)  # list로


info = mne.create_info(col_names, fs)
temp = preprocessed_data.values
temp = temp.T
s13 = mne.io.RawArray(temp, info)
#s13.plot()
#plt.show()

ch_types = ["eeg"]*28
ch_names = ["Fp1","Fp2","AF3","AF4","F7","F8","F3","Fz","F4","FC5",
            "FC6","T7","T8","C3","C4","CP5","CP6","P7","P8","P3","Pz",
            "P4","PO7","PO8","PO3","PO4","O1","O2"]
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types) # 나중에 저 친구들을 eeg로 지정해줌, 그려서 변화
temp = eeg_data.T*(1e-6) # misc에서 eeg로 변화하면서 단위 맞춰주기
montage = mne.channels.make_standard_montage("standard_1005")
s13 = mne.io.RawArray(temp, info)
s13.set_montage(montage = montage)


#################################################################################
# hight, low pass filter
#################################################################################
filtered_data = s13.filter(l_freq = 1., h_freq=50.,  picks='eeg', fir_design='firwin', skip_by_annotation='edge')
#to_data_frame(picks = 'eeg', filtered_data)
filtered_eeg = filtered_data.to_data_frame() #rawarray를 dataframe으로 변환 #이후 filtered_eeg 사용
filtered_eeg['Time (s)'] = preprocessed_data['Time (s)']
filtered_eeg=filtered_eeg.drop(['time'], axis = 1) # 자동으로 생기는 time변수 제거

filtered_df = filtered_data.to_data_frame() # filtered_df : filter되었으면서 전체 변수 다 있으면서 데이터프레임 형식.
filtered_df['TRIGGER(DIGITAL)'] = preprocessed_data['TRIGGER(DIGITAL)']
filtered_df = filtered_df.drop(['time'], axis = 1) # 자동으로 생기는 time변수 제거

#s13.plot()
#plt.show()

elec=pd.read_csv(data_path + "pos_3d_30.txt", sep='\t',header=None)
elec_xyz=elec.iloc[:,2:]
elec_xyz.iloc[:,1]=(-1)*elec_xyz.iloc[:,1]

# normalize 전극 위치
n_elec_xyz=elec_xyz/(10*elec_xyz.max())*0.75

plt.scatter(n_elec_xyz.iloc[:,1], n_elec_xyz.iloc[:,0])

# 전극의 좌우 확인
for i, txt in enumerate(elec.iloc[:,0]):
    plt.annotate(txt, ((n_elec_xyz.iloc[i,1],n_elec_xyz.iloc[i,0])))
#plt.show()

n_elec_xyz.columns=["y","x","z"]
n_elec_xyz=n_elec_xyz.iloc[:,0:2]
n_elec_xy = n_elec_xyz[['x','y']]
n_elec_xy_array=np.array(n_elec_xy)
only_eeg=eeg_data.iloc[:,0:28]

fig, ax=plt.subplots(nrows=5,ncols=5,figsize=(10, 10),sharex=True, sharey=True)
for i in range(21):
  mne.viz.plot_topomap(only_eeg.T[60*i], # 전체 채널의 특정시간 32 by 1
                      pos=n_elec_xy_array, # 전극 위치
                      axes=ax[i // 5][i % 5],
                      show=False,
                      outlines='head',
                      ch_type='eeg')
#fig.title("S13 토포그래피(단위:1분)")
#plt.show()

#################################################################################
# Comparision of IC's power spectra
#################################################################################
#define info EEG RAW
pred = filtered_eeg.iloc[:, 0:28]
sfreq = 250
ch_types = ["eeg"]*28

ch_names = ["Fp1","Fp2","AF3","AF4","F7","F8","F3","Fz","F4","FC5",
            "FC6","T7","T8","C3","C4","CP5","CP6","P7","P8","P3","Pz",
            "P4","PO7","PO8","PO3","PO4","O1","O2"]
montage = mne.channels.make_standard_montage("standard_1005")
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
samples = pred.T*1e-6
loadedRaw = mne.io.RawArray(samples, info)
loadedRaw.set_montage(montage = montage)
loadedRaw.plot_psd()
loadedRaw.plot(duration=5, n_channels=28) ##########################################여기 숫자 확인

#################################################################################
# ICA processing
#################################################################################
ica = mne.preprocessing.ICA(n_components=28, random_state=97, max_iter=800)
ica.fit(filtered_data)
ica.exclude = [0,1]
ica.plot_properties(filtered_data, picks = ica.exclude)
ica.fit(filtered_data)

filtered_data.load_data()
ica.plot_sources(filtered_data, show_scrollbars=True)
plt.show()

#orig_raw = loadedRaw.copy()
#loadedRaw.load_data()
#ica.apply(loadedRaw)

#ch_idxs = [loadedRaw.ch_names.index(ch) for ch in ch_names]
#orig_raw.plot(order = ch_idxs, start = 12, duration = 60) #########################################여기 숫자 확인
#loadedRaw.plot(order = ch_idxs, start = 12, duration = 60) #########################################여기 숫자 확인
#plt.show()


## epoch
#events = mne.make_fixed_length_events(s13, duration = 60.0)
#define info EEG RAW
#pred2 = filtered_df.copy()
#sfreq = 250
#ch_types = ["eeg"]*29

#ch_names = ["Fp1","Fp2","AF3","AF4","F7","F8","F3","Fz","F4","FC5",
#            "FC6","T7","T8","C3","C4","CP5","CP6","P7","P8","P3","Pz",
#            "P4","PO7","PO8","PO3","PO4","O1","O2",'TRIGGER(DIGITAL)']
#montage = mne.channels.make_standard_montage("standard_1005")
#info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#samples = pred2.T*1e-6
#loadedRaw2 = mne.io.RawArray(samples, info)
#loadedRaw2.set_montage(montage = montage)
#loadedRaw2.plot_psd()
#loadedRaw2.plot(duration=60, n_channels=28) ##########################################여기 숫자 확인


#events = mne.find_events(loadedRaw2, stim_channel=filtered_eeg.columns)
#print(events)


#################################################################################
# make epoch and events
#################################################################################
info = mne.create_info(col_names, fs)
temp = preprocessed_data.values
temp = temp.T
s13 = mne.io.RawArray(temp, info)
events = mne.make_fixed_length_events(s13, duration = 60)
#eeg_epoch = mne.make_fixed_length_epochs(s13, duration = 60)
epochs = mne.Epochs(s13, events)
#epochs.plot_psd(fmin = 2., fmax = 40., average = True, spatial_colors = False)
#epochs.plot_psd_topomap(ch_type = 'grade', normalize = True)

#################################################################################
# Trigger time seires
#################################################################################
plt.figure(figsize = (15,5))
plt.plot(preprocessed_data.iloc[:,0:1], preprocessed_data.iloc[:,34:35], color = "blue")
plt.xlabel('Time (sec)')
plt.ylabel('Sickness level')
plt.show()

#data_timeinclude = filtered_df.copy()
data_timeinclude = filtered_eeg.copy()


baseline = data_timeinclude.loc[data_timeinclude['Time (s)'] < 201].copy()
low_level = data_timeinclude.loc[(data_timeinclude['Time (s)'] >= 300) & (data_timeinclude['Time (s)'] < 501)].copy()
high_level = data_timeinclude.loc[(data_timeinclude['Time (s)'] >= 700) & (data_timeinclude['Time (s)'] < 901)].copy()


# baseline.rename(columns = {'Time (s)':'times'},inplace=True)
# #mne.time_frequency.csd_fourier(epochs = baseline)
# print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
# print(baseline)
#
# #picks = mne.pick_types()
#
#
#
# col_names = list(baseline.columns)
# info = mne.create_info(col_names, fs)
# temp = baseline.values
# temp = temp.T
# sts = mne.io.RawArray(temp, info)
# mne.time_frequency.psd_multitaper(eeg_epoch)

####################################################################################################
####################################################################################################
# baseline.rename(columns = {'Time (s)':'times'},inplace=True)
# #mne.time_frequency.csd_fourier(epochs = baseline)
#
# #baseline = baseline.drop(['times','ECG.(uV)', 'Resp.(Ω)', 'PPG(ADU)', 'GSR(Ω)', 'TRIGGER(DIGITAL)'], axis = 1).copy()
# col_names = list(baseline.columns)
#
# baseline['AF3(uV)'] = baseline['AF3'] - np.mean(baseline['AF3'])
# strength = np.fft.fft(baseline['AF3'])
# strength = np.log(strength) * 10
# strength = abs(strength)
# frequency = np.fft.fftfreq(len(baseline),0.02)
# plt.plot(frequency, strength)
#
# pred = baseline.iloc[:, 0:28]
# sfreq = 250
# ch_types = ["eeg"]*28
#
# ch_names = ["Fp1","Fp2","AF3","AF4","F7","F8","F3","Fz","F4","FC5",
#             "FC6","T7","T8","C3","C4","CP5","CP6","P7","P8","P3","Pz",
#             "P4","PO7","PO8","PO3","PO4","O1","O2"]
# montage = mne.channels.make_standard_montage("standard_1005")
# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# samples = pred.T*1e-6
# loadedRaw = mne.io.RawArray(samples, info)
# loadedRaw.set_montage(montage = montage)
# loadedRaw.plot_psd(fmin = 0, fmax = 40)

#################################################################################
# 앞부분과 뒷부분 남기기
#################################################################################
pred_sep = baseline[['F7','F8','F3','F4','O1','O2']]

sfreq = 250
ch_types = ["eeg"]*6

ch_names = ["F7","F8","F3","F4","O1","O2"]
montage = mne.channels.make_standard_montage("standard_1005")
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
samples = pred_sep.T*1e-6
loadedRaw = mne.io.RawArray(samples, info)
loadedRaw.set_montage(montage = montage)
loadedRaw.plot_psd(fmin = 0, fmax = 40)

