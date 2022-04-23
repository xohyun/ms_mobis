from scipy import io
import pandas as pd
import scipy
mat_file = io.loadmat('D:\\MOBIS_PJ\\ORIGINAL_DATA\\CAN_ALL\\S08_DAY1_EXPT1_DRIVE.mat')
act_accel_df = pd.DataFrame(mat_file['ACT_ACCEL'])
acc_lat_df = pd.DataFrame(mat_file['ACC_LAT'])
acc_long_df = pd.DataFrame(mat_file['ACC_LONG'])
act_break_df = pd.DataFrame(mat_file['ACT_BRAKE'])
ang_df = pd.DataFrame(mat_file['ANG'])
spd_df = pd.DataFrame(mat_file['SPD'])
yaw_rate_df = pd.DataFrame(mat_file['YAW_RATE'])

def prepro_func (start_time, end_time, df) :
    temp_df = df[round(df.iloc[:, 0], 4) >= start_time]
    prepro_df = temp_df[round(temp_df.iloc[:, 0], 4) <= end_time]
    return prepro_df

# 시간은 소수점 아래 4째자리까지 같은 것을 같은 시간으로 취급
# 처음 시작: ACT_ACCEL의 괜찮은 첫 행의 시간, 끝 시작: SPD의 처음 0 나오는 행의 시간
################### 여기 행번호만 바꿔주시면 되용~
start_time = round(act_accel_df.iloc[1339, 0], 4) # mat file의 행 시작은 1 but! python은 0부터 so, 1 빼고 사용하기
end_time = round(spd_df.iloc[68597, 0], 4)

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

scipy.io.savemat("S08_DAY1_EXPT1_preprocesse_CAN.mat",mat_file)