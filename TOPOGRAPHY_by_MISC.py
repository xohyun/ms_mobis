import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

data_path = "C:\\Users\\Dong Young\\Desktop\\PREPROCESS_DATA\\" # 데이터 경로 설정
# 보고자하는 피험자, ex) 피험자 13의 topo가 보고 싶으면, start_subject_num=13, end_subject_num=14
start_subject_num = 1
end_subject_num = 24
day = 1  # 1 or 2
expt = 2 # 1 or 2
nrow=end_subject_num//3
ncol=9
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(6,6), sharex=True, sharey=True)

# 전처리한 csv 불러오기
def data_loading(path,subject_num,day,expt):
    if subject_num < 10:  # 파일 읽기 + '\n' 제거
        preprocessed_data = pd.read_csv(path + "S0%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" % (subject_num, day, expt))
    else:
        preprocessed_data = pd.read_csv(path + "S%d_DAY%d_EXPT%d_DRIVE_preprocess.csv" % (subject_num, day, expt))

    # preprocessed_data=preprocessed_data.drop(["Unnamed: 0"],axis=1) # preprocess 다시 돌려서 나온 데이터 사용시 이부분 제거
    eeg_data = preprocessed_data.drop(
        ["Time (s)", "Packet Counter(DIGITAL)", "ECG.(uV)", "Resp.(Ω)", "PPG(ADU)", "GSR(Ω)", "Packet Counter(DIGITAL)",
         "TRIGGER(DIGITAL)"], axis=1)  # time이랑 packet counter column 제거
    eeg_col_names = list(preprocessed_data.columns)  # list로
    return preprocessed_data, eeg_data, eeg_col_names  # name에는 time이 포함되어 있음

# array형식으로 만들기
def create_array(dfdata, col_names, fs, ch_types, scale=1):
    info = mne.create_info(ch_names=col_names, sfreq=fs, ch_types=ch_types)
    temp = dfdata.values
    temp = temp.T * scale
    s_array = mne.io.RawArray(temp, info)
    return s_array

# bandpass filtering 수행하기
def bandpass_filter(arrdata, preprocessed_data):
    filtered_data = arrdata.filter(l_freq=1., h_freq=50., picks='eeg', fir_design='firwin', skip_by_annotation='edge')

    filtered_df = filtered_data.to_data_frame()  # filtered_df : filtering한 후, 전체 변수 다 있으면서 데이터프레임 형식. (time include)
    filtered_df['Time (s)'] = preprocessed_data['Time (s)']
    filtered_df = filtered_df.drop(['time'], axis=1)  # 자동으로 생기는 time변수 제거

    filtered_df_trigger = filtered_df.copy()
    filtered_df_trigger['TRIGGER(DIGITAL)'] = preprocessed_data['TRIGGER(DIGITAL)']
    filtered_df_trigger=filtered_df_trigger.drop(['Time (s)'], axis=1)  # 자동으로 생기는 time변수 제거
    return filtered_data, filtered_df, filtered_df_trigger

# plot 보여주는 함수
def show_plot(arrdata):
    arrdata.plot()
    plt.show()

# 사용한 전극의 위치
def electrode_pos():
    elec=pd.read_csv("pos_3d_30.txt", sep='\t',header=None)
    elec_xyz=elec.iloc[:,2:]
    elec_xyz.iloc[:,1]=(-1)*elec_xyz.iloc[:,1]

    # normalize 전극 위치
    n_elec_xyz=elec_xyz/(10*elec_xyz.max())*0.75

    n_elec_xyz.columns=["y","x","z"]
    n_elec_xyz=n_elec_xyz.iloc[:,0:2]
    n_elec_xy = n_elec_xyz[['x','y']]
    elec_2d_pos_array=np.array(n_elec_xy)
    return elec_2d_pos_array

def divide_data(df,level):
    # Divde by MISC (normal:0~1, low:2~5, high:6~9)
    normal = df.loc[(df["TRIGGER(DIGITAL)"] >= level[0]) & (df["TRIGGER(DIGITAL)"] < level[1])]
    low = df.loc[(df["TRIGGER(DIGITAL)"] >= level[1]) & (df["TRIGGER(DIGITAL)"] < level[2])]
    high = df.loc[(df["TRIGGER(DIGITAL)"] >= level[2]) & (df["TRIGGER(DIGITAL)"] < 10)]

    # mean,delete trigger -> Dataframe
    notrigger_normal = normal.mean().to_frame().drop(["TRIGGER(DIGITAL)"], axis=0)
    notrigger_low = low.mean().to_frame().drop(["TRIGGER(DIGITAL)"], axis=0)
    notrigger_high = high.mean().to_frame().drop(["TRIGGER(DIGITAL)"], axis=0)

    return [notrigger_normal,notrigger_low,notrigger_high]

def plot_topo(sbj_num,eeg_data,elec_pos,row,col):
    if nrow==1:
        sub_pos=ax[col]
    else:
        sub_pos=ax[row,col]
    mne.viz.plot_topomap(eeg_data[0],
                        pos=elec_pos,  # 전극 위치
                        axes=sub_pos,
                        show=False, outlines='head', ch_type='eeg')

    # N: normal, L: low, H: high
    if col%3==0:
        sub_pos.set_title("S%d N"%(sbj_num),fontsize=11)
    elif col%3==1:
        sub_pos.set_title("S%d L"%(sbj_num),fontsize=11)
    else:
        sub_pos.set_title("S%d H"%(sbj_num),fontsize=11)


def main():
    elec_2d_pos_array=electrode_pos()
    row = 0
    col = 0

    for sbj_num in range (start_subject_num,end_subject_num):
        preprocessed_data, eeg_data, eeg_col_names = data_loading(data_path, sbj_num, day, expt)
        ch_names = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8", "F3", "Fz", "F4", "FC5",
                    "FC6", "T7", "T8", "C3", "C4", "CP5", "CP6", "P7", "P8", "P3", "Pz",
                    "P4", "PO7", "PO8", "PO3", "PO4", "O1", "O2"]  # 채널 이름 설정
        ch_types = ["eeg"] * 28
        fs=250
        s_array = create_array(eeg_data, ch_names, fs, ch_types, scale=(1e-6))
        # show_plot(s_array)

        montage = mne.channels.make_standard_montage("standard_1005")
        s_array.set_montage(montage=montage)

        filtered_arr, filter_df, filter_df_trigger = bandpass_filter(s_array, preprocessed_data)
        # show_plot(filtered_arr)

        # Divde by MISC (normal:0~1, low:2~5, high:6~9)
        MISC_based_divide_data_list = divide_data(filter_df_trigger,[0,2,6]) # [normal,low,high]

        # normal, low, high topography
        for i in range(3):
            plot_topo(sbj_num,MISC_based_divide_data_list[i],elec_2d_pos_array,row,col+i)

        # move to next subplot
        if (col < 6) and (sbj_num % 3 != 0):
            col += 3
        if (col == 6) and (sbj_num % 3 == 0):
            row+=1
            col=0

if __name__ == "__main__":
    main()
    plt.tight_layout()
    plt.show()

