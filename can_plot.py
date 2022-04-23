from scipy import io
import pandas as pd
import scipy
import matplotlib as mpl
import matplotlib.pylab as plt

for i in range(16,17) : #17,24
    for j in range(1,3) :
        for k in range(1, 3) :
            mat_file = io.loadmat('C:\\Users\\soso\\Desktop\\data\data_original\\CAN_ALL.mat' % (i, j, k))
            #mat_file = io.loadmat('C:\\Users\\soso\\Desktop\\can_pre\\S%d_DAY%d_EXPT%d_preprocesse_CAN.mat' % (i, j, k))
            act_accel_df = pd.DataFrame(mat_file['ACT_ACCEL'])
            plt.plot(act_accel_df[0], act_accel_df[1])
            plt.show()