import pandas as pd

data_path = "ORIGINAL_DATA\\TOTAL_DATA\\" # 원본 데이터 경로 설정
labeling_drive_path = "LABELING_DRIVE\\" # 전처리 후 데이터 경로 설정
label_path = "LABELS\\" # LABEL 데이터 경로 설정

# 라벨링 하고자하는 피험자 EX) 1번부터 13번 -> start_subject_num=1, end_subject_num=13
start_subject_num = 1
end_subject_num = 23

# 데이터 불러오기
def data_loading(path,subject_num,day,expt):
    if subject_num < 10:  # 파일 읽기 + '\n' 제거
        with open(path+'S0%d_DAY%d_EXPT%d_DRIVE.txt' % (subject_num,day,expt), "r", encoding='utf-8') as f:
            original_data = f.read().splitlines()
    else:
        with open(path+'S%d_DAY%d_EXPT%d_DRIVE.txt' % (subject_num,day,expt), "r", encoding='utf-8') as f:
            original_data = f.read().splitlines()
    original_data = original_data[4:] # 필요 없는 처음 4줄 제거
    return original_data

# 실험 중에 수집한 데이터만 추출
def cut_experimenting_data(data,sbj_num,day,expt):
    a = []
    for i in range(len(data)):
        b = data[i].replace(",", "").rstrip("\t").split("\t")  # ,와 마지막의 tab 지우기
        if len(b) == 49:
            a.append(b[0:49])  # 한 row씩 list에 저장(2차원 list), column 개수는 50개

    trigger_idx = []  # TRIGGER가 있는 index 찾기 for 실험 시작과 끝 자르기
    for i in range(1, len(a)):  # 첫번째 행은 column이름들
        if a[i][48] != '':  # TRIGGER(DIGITAL) 까지의 column
            if float(a[i][48]) != 0.0: # 0이 아닌 TRIGGER의 index 수집
                trigger_idx.append(i)

    # END TIME 입력, 마지막에서 speed가 0이 되는 첫번째 시간
    if (day==1) and (expt == 1):   # S01,04
        if sbj_num == 1:  # 1603.450855961000s -> 1603.454s -> 1543.45s
            trigger_idx.append(771725)
        elif sbj_num == 4:  # 1308.514932753000s -> 1308.516s -> 1248.516s -> 1221.95s
            trigger_idx.append(610974)
    elif (day == 1) and (expt == 2): # S08
        if sbj_num == 8: # 22분 까지 유효
            trigger_idx=trigger_idx[:23]
    elif (day == 2) and (expt == 1): # S04,10,19
        if sbj_num == 4:  # 1493.117165864000s -> 1493.118s -> 1433.118s
            trigger_idx.append(716559)
        elif sbj_num == 10:  # 1411.927169381000 -> 1411.946s -> 1351.946s
            trigger_idx.append(675973)
        elif sbj_num == 19:  # 1332.792648517000s -> 1332.794s -> 1292.794s
            trigger_idx.append(646397)

    total_data_df = pd.DataFrame(a, columns=a[0]) # dataframe으로 변화
    cut_expt_df = total_data_df.loc[trigger_idx[0]: trigger_idx[-1]] # 실험 시작 ~ 실험 끝

    # 사용하는 column 선택
    cut_expt_df = cut_expt_df[['Time (s)', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)', 'AF4(uV)', 'F7(uV)', 'F8(uV)',
                       'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
                       'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)', 'P8(uV)', 'P3(uV)',
                       'Pz(uV)', 'P4(uV)', 'PO7(uV)', 'PO8(uV)', 'PO3(uV)', 'PO4(uV)', 'O1(uV)',
                       'O2(uV)', 'ECG.(uV)', 'Resp.(Ω)', 'PPG(ADU)', 'GSR(Ω)', 'Packet Counter(DIGITAL)',
                       'TRIGGER(DIGITAL)']]
    expt_df = cut_expt_df.astype(float)  # 문자형을 float으로 변환
    return expt_df, trigger_idx

# 추가로 LABEL 수정
def additional_trigger_change(trigger, sbj_num,day,expt):
    if (day == 1) and (expt == 1) :  # S02,05,12,20
        if sbj_num == 2:
            trigger[15] = 4.0
            trigger[16] = 4.0
        elif sbj_num == 5:
            trigger[7] = 3.0
        elif sbj_num == 12:
            trigger[9] = 2.0
        elif sbj_num == 20:
            trigger[9] = 4.0
            trigger[10] = 4.0
            trigger[11] = 4.0
    elif (day == 1) and (expt == 2): # S01,02,06,07,10,12,13,16
        if sbj_num == 1:
            trigger[7] = 3.0
            trigger[12] = 3.0
            trigger[21] = 4.0
        elif sbj_num == 2:
            trigger[6] = 1.0
        elif sbj_num == 6:
            trigger[11] = 6.0
        elif sbj_num == 7:
            trigger[22] = 3.0
        elif sbj_num == 10:
            trigger[9] = 4.0
            trigger[11] = 5.0
            trigger[15] = 6.0
        elif sbj_num == 12:
            trigger[8] = 2.0
        elif sbj_num == 13:
            trigger[20] = 6.0
            trigger[21] = 6.0
        elif sbj_num == 16:
            trigger[17] = 6.0
    elif (day == 2) and (expt == 1): # S01,07,10,13
        if sbj_num == 1:
            trigger[18] = 2.0
            trigger[19] = 2.0
        elif sbj_num == 7:
            trigger[22] = 2.0
        elif sbj_num == 10:
            trigger[17]=5.0
            trigger[18:]=[6.0]*(len(trigger)-16)
        elif sbj_num == 13:
            trigger[10] = 6.0
    elif (day == 2) and (expt == 2): # S01,16,19
        if sbj_num == 1:
            trigger[17] = 3.0
        elif sbj_num == 16:
            trigger[18] = 5.0
        elif sbj_num == 19:
            trigger[1] = 0.0
            trigger[-1] = 3.0
    return trigger

# 데이터 저장
def data_save(path,subject_num,day,expt,data,type):
    if subject_num < 10:
        data.to_csv(path+"S0%d_DAY%d_EXPT%d_DRIVE_%s.csv" % (subject_num,day,expt,type),index=False, columns=data.columns)
    else:
        data.to_csv(path + "S%d_DAY%d_EXPT%d_DRIVE_%s.csv" % (subject_num,day,expt,type), index=False, columns=data.columns)

def main(day,expt):
    for sbj_num in range (start_subject_num,end_subject_num+1):
        print("[피험자 %d]" % (sbj_num))
        original_data= data_loading(data_path, sbj_num, day, expt) # 데이터 불러오기

        expt_df,original_trigger_idx = cut_experimenting_data(original_data,sbj_num,day,expt) # 실험 중에 얻은 데이터 추출
        ori_start_idx = original_trigger_idx[0]  # 시작 9를 입력한 idx
        ori_end_idx = original_trigger_idx[-1]  # 마지막 TRIGGER를 입력한 idx
        original_trigger = list(expt_df.loc[original_trigger_idx, "TRIGGER(DIGITAL)"])  # TRIGGER 저장
        print("original trigger idx:",original_trigger_idx)
        print("original trigger:",original_trigger)

        ### 각 TRIGGER가 찍힌 시간에 맞춰서 TRIGGER 작성
        # 마지막 trigger를 찍은 시간에 따라 TRIGGER를 입력할 리스트 초기화
        if (original_trigger_idx[-1]//30000- original_trigger_idx[-2]//30000) > 0:
            change_trigger=[0.0]*(original_trigger_idx[-1]//30000)
        else:
            change_trigger = [0.0] * (original_trigger_idx[-1] // 30000 + 1)

        for i in original_trigger_idx[:-1]:
            if change_trigger[i//30000]==0:
                change_trigger[i//30000]=expt_df.loc[i][-1]
        change_trigger[0] = 0.0  # 첫번째 TRIGGER를 0으로 변화
        change_trigger.append(original_trigger[-1]) # 마지막 trigger

        # 추가로 TRIGGER 변화
        if (sbj_num==1) or (sbj_num==2) or (sbj_num==5) or (sbj_num==6) or (sbj_num==7) or (sbj_num==10) or (
                sbj_num==12) or (sbj_num==13) or (sbj_num==16) or (sbj_num==19) or (sbj_num==20):
            change_trigger = additional_trigger_change(change_trigger,sbj_num,day,expt)
        print("changed trigger:", change_trigger)

        ### 정한 규칙에 따라 TRIGGER 입력 (-30s~30s), 시작점 index를 0으로 바꿈
        change_trigger_idx = [0] + list(range(30000, ori_end_idx - ori_start_idx + 1, 30000))
        change_trigger_idx[1:]=[x-15000 for x in change_trigger_idx[1:]] # 30초 당기기

        expt_df.iloc[0:15000, -1] = change_trigger[0]  # 첫번째 TRIGGER는 30초만 삽입
        for i in change_trigger_idx[1:]:  # 첫번째 이후
            if i != change_trigger_idx[-1]: # TRIGGER들을 1분씩 입력
                expt_df.iloc[i:i + 30000, -1] = change_trigger[i//30000+1]
            else:
                # 마지막에서 두번째의 TRIGGER를 데이터 끝까지 입력
                expt_df.iloc[i:, -1] = change_trigger[i//30000+1]

        chg_end_idx=ori_end_idx-ori_start_idx
        chg_second_end_idx = change_trigger_idx[-1]

        # 마지막 현재 상태 입력과 바로 전 입력 사이에 차이에 따라 끝부분 수정
        if chg_end_idx-chg_second_end_idx<30000:
            # EX) 마지막에서 두번째: 5를 5분 10초에 입력, 마지막 6을 5분 25초에 입력 -> 4분30초~5분25초에 5 입력, 5분25초에 6 입력
            expt_df.iloc[-1, -1] = change_trigger[-1]  # 맨 마지막에 TRIGGER 삽입
        else:
            # EX) 마지막에서 두번째: 5를 5분 10초에 입력 -> 4분30초~5분30초에 5 입력,
            # 마지막 6을 5분 40초에 입력한 경우 -> 5분 30초~5분40초에 6 입력
            expt_df.iloc[change_trigger_idx[-1]+30000:,-1] = change_trigger[-1]
            change_trigger_idx.append(change_trigger_idx[-1]+30000)
        change_trigger_idx.append(chg_end_idx) # 마지막 TRIGGER의 IDX 추가

        data_save(labeling_drive_path,sbj_num,day,expt,expt_df,"preprocess") # 데이터 저장

        # LABEL 추출 및 저장
        trigger_data = expt_df.iloc[change_trigger_idx, [0,-1]]
        data_save(label_path, sbj_num, day, expt, trigger_data, "labels")

if __name__ == '__main__':
    for day in range(2,3):
        for expt in range(2,3):
            print("----------DAY%d EXPT%d----------"%(day,expt))
            main(day,expt)
