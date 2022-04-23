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

def subj_preprocessing(subj_num) :
    mat_cfile_1 = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S0%d_DAY1_EXPT1_DRIVE_1.mat'%subj_num)
    mat_cfile_copy = mat_cfile_1.copy()
    mat_cfile_2 = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S0%d_DAY1_EXPT1_DRIVE_2.mat'%subj_num)
    mat_cfile_12 = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S0%d_DAY2_EXPT1_DRIVE.mat'%subj_num)

    act_accel_df1, acc_lat_df1, acc_long_df1, act_break_df1, ang_df1, spd_df1, yaw_rate_df1 = mat_to_df(mat_cfile_1)
    act_accel_df2, acc_lat_df2, acc_long_df2, act_break_df2, ang_df2, spd_df2, yaw_rate_df2 = mat_to_df(mat_cfile_2)
    act_accel_df_c, acc_lat_df_c, acc_long_df_c, act_break_df_c, ang_df_c, spd_df_c, yaw_rate_df_c = mat_to_df(mat_cfile_12)

    # a = 14672
    # b = 24722

    a = 54429 #subj6 
    b = 61020 

    temp_act_accel = act_accel_df_c.iloc[(2*a):(2*b)].copy()
    temp_acc_lat = acc_lat_df_c.iloc[(2*a):(2*b)].copy()
    temp_acc_long = acc_long_df_c.iloc[(2*a):(2*b)].copy()
    temp_acc_break = act_break_df_c.iloc[a:b].copy()
    temp_acc_ang = ang_df_c.iloc[(2*a):(2*b)].copy()
    temp_spd = spd_df_c.iloc[a:b].copy()
    temp_yaw_rate = yaw_rate_df_c.iloc[(2*a):(2*b)].copy()

    # c = 34.8217
    # d = 459.6202
    c = 27.0917
    d = 1220.3915
    
    temp_act_accel[0] -= c
    temp_acc_lat[0] -= c
    temp_acc_long[0] -= c
    temp_acc_break[0] -= c
    temp_acc_ang[0] -= c
    temp_spd[0] -= c
    temp_yaw_rate[0] -= c

    act_accel_df2[0] += d
    acc_lat_df2[0] += d
    acc_long_df2[0] += d
    act_break_df2[0] += d
    ang_df2[0] += d
    spd_df2[0] += d
    yaw_rate_df2[0] += d

    act_accel_df_c = pd.concat([act_accel_df1, temp_act_accel, act_accel_df2])
    acc_lat_df_c = pd.concat([acc_lat_df1, temp_acc_lat, acc_lat_df2])
    acc_long_df_c = pd.concat([acc_long_df1, temp_acc_long, acc_long_df2])
    act_break_df_c = pd.concat([act_break_df1, temp_acc_break, act_break_df2])
    ang_df_c = pd.concat([ang_df1, temp_acc_ang, ang_df2])
    spd_df_c = pd.concat([spd_df1, temp_spd, spd_df2])
    yaw_rate_df_c = pd.concat([yaw_rate_df1, temp_yaw_rate, yaw_rate_df2])

    mat_cfile_copy['ACC_LAT'] = acc_lat_df_c.values.tolist()
    mat_cfile_copy['ACC_LONG'] = acc_long_df_c.values.tolist()
    mat_cfile_copy['ACT_ACCEL'] = act_accel_df_c.values.tolist()
    mat_cfile_copy['ACT_BRAKE'] = act_break_df_c.values.tolist()
    mat_cfile_copy['ANG'] = ang_df_c.values.tolist()
    mat_cfile_copy['SPD'] = spd_df_c.values.tolist()
    mat_cfile_copy['YAW_RATE'] = yaw_rate_df_c.values.tolist()

    scipy.io.savemat("E:\\data\data_original\\CAN_ALL\\S0%d_DAY1_EXPT1_DRIVE.mat"%subj_num, mat_cfile_copy, do_compression = True)

#se_tbl = [[1049,65377]] # subject05 day1 expt1
se_tbl = [[390,67700]] # subject06 day1 expt1

# 시간은 소수점 아래 4째자리까지 같은 것을 같은 시간으로 취급
# 처음 시작: ACT_ACCEL의 괜찮은 첫 행의 시간, 끝 시작: SPD의 처음 0 나오는 행의 시간
################### 여기 행번호만 바꿔주시면 되용~
subj_preprocessing(6)
subj = [6] #16,17,18,19,20,21,22,23
for i in range(len(subj)) :
    for j in range(1, 2) :
        for k in range(1, 2) :
            print("%d %d %d" % (subj[i], j, k))

            mat_file = io.loadmat('E:\\data\\data_original\\CAN_ALL\\S0%d_DAY%d_EXPT%d_DRIVE.mat' % (subj[i], j, k))
            act_accel_df, acc_lat_df, acc_long_df, act_break_df, ang_df, spd_df, yaw_rate_df = mat_to_df(mat_file)

            start_time = round(spd_df.iloc[se_tbl[i][0] -251, 0], 4) # mat file의 행 시작은 1 but! python은 0부터 so, 1 빼고 사용하기
            if (se_tbl[i][1] + 250) > len(spd_df) :
                temp2 = len(spd_df)
            else :
                print("*******************************%d %d %d" % (subj[i], j, k))
                temp2 = se_tbl[i][1] + 250 #250
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

            scipy.io.savemat("C:\\Users\\soso\\Desktop\\can_pre\\S0%d_DAY%d_EXPT%d_preprocess_CAN.mat" % (subj[i], j, k), mat_file, do_compression = True)


# start_time = round(spd_df.iloc[1262-251, 0], 4) # mat file의 행 시작은 1 but! python은 0부터 so, 1 빼고 사용하기
# end_time = round(spd_df.iloc[73927+250, 0], 4)

# scipy.io.savemat("C:\\Users\\soso\\Desktop\\can_pre\\S18_DAY1_EXPT1_preprocess_CAN.mat",mat_file)