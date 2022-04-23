from scipy import io
import pandas as pd
import scipy
import numpy

def prepro_func (start_time, end_time, df) :
    temp_df = df[round(df.iloc[:, 0], 4) >= start_time]
    prepro_df = temp_df[round(temp_df.iloc[:, 0], 4) <= end_time]
    return prepro_df

def mat_to_df (mat_file) :
    act_accel_df = pd.DataFrame(mat_file['ACT_ACCEL'])
    acc_lat_df = pd.DataFrame(mat_file['ACC_LAT'])
    acc_long_df = pd.DataFrame(mat_file['ACC_LONG'])
    act_break_df = pd.DataFrame(mat_file['ACT_BRAKE'])
    ang_df = pd.DataFrame(mat_file['ANG'])
    spd_df = pd.DataFrame(mat_file['SPD'])
    yaw_rate_df = pd.DataFrame(mat_file['YAW_RATE'])
    return act_accel_df, acc_lat_df, acc_long_df, act_break_df, ang_df, spd_df, yaw_rate_df

def subj16_preprocessing () :
    mat_16file_1 = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S16_DAY2_EXPT2_DRIVE_1.mat')
    mat_16file_2 = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S16_DAY2_EXPT2_DRIVE_2.mat')
    mat_16file_12 = io.loadmat('C:\\Users\\soso\\Desktop\\can_pre\\S16_DAY1_EXPT2_preprocess_CAN.mat')

    act_accel_df1, acc_lat_df1, acc_long_df1, act_break_df1, ang_df1, spd_df1, yaw_rate_df1 = mat_to_df(mat_16file_1)
    act_accel_df2, acc_lat_df2, acc_long_df2, act_break_df2, ang_df2, spd_df2, yaw_rate_df2 = mat_to_df(mat_16file_2)
    act_accel_df_c, acc_lat_df_c, acc_long_df_c, act_break_df_c, ang_df_c, spd_df_c, yaw_rate_df_c = mat_to_df(mat_16file_12)

    temp_act_accel = act_accel_df_c.iloc[14208:16910].copy()
    temp_acc_lat = acc_lat_df_c.iloc[14208:16910].copy()
    temp_acc_long = acc_long_df_c.iloc[14208:16910].copy()
    temp_acc_break = act_break_df_c.iloc[7104:8455].copy()
    temp_acc_ang = ang_df_c.iloc[14208:16910].copy()
    temp_spd = spd_df_c.iloc[7104:8455].copy()
    temp_yaw_rate = yaw_rate_df_c.iloc[14208:16910].copy()

    temp_act_accel[0] += 183.1158
    temp_acc_lat[0] += 183.1158
    temp_acc_long[0] += 183.1158
    temp_acc_break[0] += 183.1158
    temp_acc_ang[0] += 183.1158
    temp_spd[0] += 183.1158
    temp_yaw_rate[0] += 183.1158

    act_accel_df2[0] += 360.2240
    acc_lat_df2[0] += 360.2240
    acc_long_df2[0] += 360.2240
    act_break_df2[0] += 360.2240
    ang_df2[0] += 360.2240
    spd_df2[0] += 360.2240
    yaw_rate_df2[0] += 360.2240

    act_accel_df_16 = pd.concat([act_accel_df1, temp_act_accel, act_accel_df2])
    acc_lat_df_16 = pd.concat([acc_lat_df1, temp_acc_lat, acc_lat_df2])
    acc_long_df_16 = pd.concat([acc_long_df1, temp_acc_long, acc_long_df2])
    act_break_df_16 = pd.concat([act_break_df1, temp_acc_break, act_break_df2])
    ang_df_16 = pd.concat([ang_df1, temp_acc_ang, ang_df2])
    spd_df_16 = pd.concat([spd_df1, temp_spd, spd_df2])
    yaw_rate_df_16 = pd.concat([yaw_rate_df1, temp_yaw_rate, yaw_rate_df2])

    mat_16file_1['ACC_LAT'] = acc_lat_df_16.values.tolist()
    mat_16file_1['ACC_LONG'] = acc_long_df_16.values.tolist()
    mat_16file_1['ACT_ACCEL'] = act_accel_df_16.values.tolist()
    mat_16file_1['ACT_BRAKE'] = act_break_df_16.values.tolist()
    mat_16file_1['ANG'] = ang_df_16.values.tolist()
    mat_16file_1['SPD'] = spd_df_16.values.tolist()
    mat_16file_1['YAW_RATE'] = yaw_rate_df_16.values.tolist()

    scipy.io.savemat("E:\\data\data_original\\CAN_ALL\\S16_DAY2_EXPT2_DRIVE.mat", mat_16file_1, do_compression = True)

# 17~23의 전처리 / 순서대로 17_day1_expt1, 17_day1_expt2,...
se_tbl = [[651,72207],[1259,65474],[1183,66619],[548,62224],
        [1262,73927],[752,63753],[1112,68353],[1329,62959],
        [914,67033],[768,62755],[762,66592],[3238,65758],
        [918,80731],[1690,64321],[3234,70476],[567,64976],
        [2259,72477],[935,63503],[884,66500],[881,64550],
        [1191,71453],[737,63199],[562,67429],[2199,64600],
        [658,66748],[1543,64303],[725,64984],[1905,62649]
]

# 시간은 소수점 아래 4째자리까지 같은 것을 같은 시간으로 취급
# 처음 시작: ACT_ACCEL의 괜찮은 첫 행의 시간, 끝 시작: SPD의 처음 0 나오는 행의 시간
################### 여기 행번호만 바꿔주시면 되용~
subj16_preprocessing()
subj = [16] #16,17,18,19,20,21,22,23
for i in range(len(subj)) :
    for j in range(2, 3) :
        for k in range(2, 3) :
            print("%d %d %d" % (subj[i], j, k))

            mat_file = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S%d_DAY%d_EXPT%d_DRIVE.mat' % (subj[i], j, k))
            act_accel_df, acc_lat_df, acc_long_df, act_break_df, ang_df, spd_df, yaw_rate_df = mat_to_df(mat_file)

            start_time = round(spd_df.iloc[se_tbl[i][0] -251, 0], 4) # mat file의 행 시작은 1 but! python은 0부터 so, 1 빼고 사용하기
            if (se_tbl[i][1] + 250) > len(spd_df) :
                temp2 = len(spd_df)
            else :
                print("*******************************%d %d %d" % (subj[i], j, k))
                temp2 = se_tbl[i][1] + 300 #250
            end_time = round(temp2, 4)

            acc_lat_df_prepro = prepro_func(start_time, end_time, acc_lat_df)
            act_accel_df_prepro = prepro_func(start_time, end_time, act_accel_df)
            acc_long_df_prepro = prepro_func(start_time, end_time, acc_long_df)
            act_break_df_prepro = prepro_func(start_time, end_time, act_break_df)
            ang_df_prepro = prepro_func(start_time, end_time, ang_df)
            spd_df_prepro = prepro_func(start_time, end_time, spd_df)
            yaw_rate_df_prepro = prepro_func(start_time, end_time, yaw_rate_df)

            mat_file['ACC_LAT'] = acc_lat_df_prepro.values.tolist()
            mat_file['ACC_LONG'] = acc_long_df_prepro.values.tolist()
            mat_file['ACT_ACCEL'] = act_accel_df_prepro.values.tolist()
            mat_file['ACT_BRAKE'] = act_break_df_prepro.values.tolist()
            mat_file['ANG'] = ang_df_prepro.values.tolist()
            mat_file['SPD'] = spd_df_prepro.values.tolist()
            mat_file['YAW_RATE'] = yaw_rate_df_prepro.values.tolist()

            scipy.io.savemat("C:\\Users\\soso\\Desktop\\can_pre\\S%d_DAY%d_EXPT%d_preprocess_CAN.mat" % (subj[i], j, k), mat_file, do_compression = True)


# start_time = round(spd_df.iloc[1262-251, 0], 4) # mat file의 행 시작은 1 but! python은 0부터 so, 1 빼고 사용하기
# end_time = round(spd_df.iloc[73927+250, 0], 4)

# scipy.io.savemat("C:\\Users\\soso\\Desktop\\can_pre\\S18_DAY1_EXPT1_preprocess_CAN.mat",mat_file)