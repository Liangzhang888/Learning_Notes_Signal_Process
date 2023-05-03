import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import  matplotlib as mpl
import os


mpl.use('TKAgg')
#不使用这个的话，PyCharm可能会由bug
# Reference:https://blog.csdn.net/qq_51248682/article/details/129396134

def GenerateData(path):
    #生成数据
    eeg_signal = np.loadtxt(path)
    return eeg_signal
def EMDFunction(signal):
    emd = EMD()
    #对EEG信号进行emd分解
    imfs = emd(signal,max_imf=6)

    #绘图
    plt.figure(figsize=(7,30))
    plt.subplot(len(imfs) + 1, 1, 1)
    # subplot(a,b,c) 参数a,b,c分别表示 行 列 图像的位置(1,2,3,...)
    plt.plot(signal)
    plt.title('Raw Data')

    for i ,imf in enumerate(imfs):
        plt.subplot(len(imfs)+1,1,i+2)
        plt.plot(imf)
        plt.title('IMF{}'.format(i+1))

    plt.tight_layout()
    plt.show()
def main():
    #获取path
    path = "D:\\PythonCode\\FirstPaper_SDP\\UBonn_eegdatasets_sampled\\A_Z\\Z001.txt"
    EEG_data = GenerateData(path)

    # 切片
    # EEG_data = EEG_data[:250]
    # print(np.array(EEG_data).shape)

    EMDFunction(EEG_data)

if __name__ == "__main__":
    main()


