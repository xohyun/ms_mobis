import numpy as np
import pandas as pd

with open('C:\\Users\\soso\\Desktop\\data\\BIO_S13_S15\\S13_DAY1_EXPT2_DRIVE.txt', "r") as f:
    original_data = f.readlines()

original_data = original_data[4:]
a = []
for i in range(len(original_data)):
    b = original_data[i].replace(",","").split("\t")
    a.append(b[0:len(b)-1])

clean_tab = pd.DataFrame(np.zeros((len(a), len(b)-1)))

for i in range(len(a)):
        print(i)
        clean_tab.loc[i]=a[i]

# 사용할 column들 선택
clean_tab.columns = clean_tab.loc[0]
clean_tab = clean_tab[1:]
clean_tab = clean_tab[['Time (s)', 'Fp1(uV)', 'Fp2(uV)', 'AF3(uV)', 'AF4(uV)', 'F7(uV)', 'F8(uV)',
           'F3(uV)', 'Fz(uV)', 'F4(uV)', 'FC5(uV)', 'FC6(uV)', 'T7(uV)', 'T8(uV)',
           'C3(uV)', 'C4(uV)', 'CP5(uV)', 'CP6(uV)', 'P7(uV)', 'P8(uV)', 'P3(uV)',
           'Pz(uV)', 'P4(uV)', 'PO7(uV)', 'PO8(uV)', 'PO3(uV)', 'PO4(uV)', 'O1(uV)',
           'O2(uV)','ECG.(uV)', 'Resp.(Ω)', 'PPG(ADU)', 'GSR(Ω)', 'Packet Counter(DIGITAL)',
           'TRIGGER(DIGITAL)']]
newdata = clean_tab.astype(float)
#문자형을 float으로 변환
newdata.loc[newdata.iloc[:,-1]!=0]

idxs = newdata.loc[newdata['TRIGGER(DIGITAL)'] != 0].index
start_idx = idxs[0]
end_idx = idxs[len(idxs)-1]

cutting_df = newdata.loc[start_idx : end_idx]

cutting_df.to_csv("cut_data.csv", index = False, columns = cutting_df.columns)

odf= pd.read_csv("cut_data.csv")
odf.head()

trigger=list(odf.iloc[odf[odf.iloc[:,-1]!=0].index,-1])
base_time=list(odf.iloc[odf[odf.iloc[:,-1]!=0].index,0])[0] #start time

trigger[0]=0.0 # 첫번째 trigger를 0으로 변환
del trigger[14] # 1분 내의 중복된 trigger 제거
trigger.append(6) # 생략된 trigger 추가

cdf=odf # 복사
cdf.iloc[:,-1]=0 # trigger를 모두 0을 초기화

for i in range(len(trigger)):
  idx=cdf[cdf.iloc[:,0]==base_time+60*i].index
  cdf.iloc[idx,-1]=trigger[i]

cdf[cdf.iloc[:,-1]!=0].index  # index를 확인한 결과 30000 단위로 나누어짐
trigger_double=np.repeat(trigger,30000) # trigger 삽입을 위한 cdf와 길이 맞추기

for i in range(len(cdf)):
  cdf.iloc[i,-1]=trigger_double[i] # trigger 삽입

cdf.to_csv("before_downsampling.csv",index = False, columns = cdf.columns)

perfect_data = pd.DataFrame(np.zeros((len(cdf)//50, len(cdf.columns))),columns=cdf.columns)
for i in range((len(cdf)+1)//50):
  perfect_data.loc[i]=cdf.loc[i*50:50*(i+1)].mean()

perfect_data.to_csv("preprocessed_data_average.csv",index = False, columns = perfect_data.columns) # 저장

idxs=range(0,len(cdf),50)
final_data = cdf.loc[idxs]

final_data.to_csv("preprocessed_data_by0.002s.csv",index = False, columns = final_data.columns) # 저장

#import matplotlib.pyplot as plt

#plt.plot(final_data)
#plt.legend()
#plt.show()















