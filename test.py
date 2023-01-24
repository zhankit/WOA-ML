#
# # test = 'infant1_ecg'
# # data_dir = "../tset01/physionet.org/files/picsdb/1.0.0"
# #
# # for i in range(10):
# #     test = 'infant{}_ecg'.format(i)
# #     print(test)
# #     print('this is the test {}'.format(i))
#
# import wget
# import wfdb
# import matplotlib.pyplot as plt
# import numpy as np
#
# data_dir = "../tset01/physionet.org/files/picsdb/1.0.0"
#
# time_arr = np.arange(0, 82122000, 1/250)
# print(time_arr)
#
# # for i in range(10):
# #     infant_ecg='infant{}_ecg'.format(i+1)
# #     loadpath= data_dir + '/' + infant_ecg
# #     ecg_record = wfdb.rdsamp(loadpath)
# #     plt.plot(ecg_record[0][:500])
# #     plt.show()
# #     i=i+1
# #     print('this is the {} data'.format(i))
# #     print("________________________________")
#
#
#
#
# # print(ecg_record)
# # print("-------------------------------------")
# # print(ecg_record[1]['fs'])
# #
# # resp_record = wfdb.rdsamp(f"{data_dir}/infant1_resp")
# #
# # time = len(resp_record[0]) * resp_record[1]['fs'] / 60
# # time_arr = np.arange(0, len(resp_record[0]), 1/resp_record[1]['fs'])
# #
# # print(time_arr)
# # plt.plot(resp_record[0][100000:101000])
# # plt.show()
# #
# #
# # print(resp_record)
#

import math
import numpy as np
import pywt
import argparse
import matplotlib.pyplot as plt
import wfdb
from scipy import signal as sig


## jerry 20211129
## 对ECG采集器采到的数据滤波，定位QRS波，计算心率
## 命令示例：python ./ecg_basics.py -e "D:/code/python/ECG/ld_data/test/163.dat"

def draw(data):
    plt.figure(figsize=(16, 2))
    plt.plot(data)
    plt.show()


# 滤-肌电干扰、工频干扰
def filter_01(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 滤-基线漂移
def filter_02(data, fs=500):
    data = np.array(data)
    winsize = int(round(0.2 * fs))
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = sig.medfilt(data, kernel_size=winsize)
    winsize = int(round(0.6 * fs))
    if winsize % 2 == 0:
        winsize += 1
    baseline_estimate = sig.medfilt(baseline_estimate, kernel_size=winsize)
    ecg_blr = data - baseline_estimate
    return ecg_blr.tolist()


# QRS波定位
def getQRS(data):
    count_q = []
    count_r = []
    count_s = []

    T = 256
    N = 24
    rE = T // 3
    E = T // 7
    x = data[:]
    x = x.astype("float")
    x = (x - np.mean(x)) / np.std(x)

    # plt.figure(num=None, figsize=(16, 2), dpi=80)

    x1 = sig.lfilter([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1], [1, -2, 1], x)
    x2 = sig.lfilter(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1],
        x1)

    # plt.plot(x2[0:3000])

    x3 = np.zeros(x.shape)
    for i in range(2, len(x2) - 2):
        x3[i] = (-1 * x2[i - 2] - 2 * x2[i - 1] + 2 * x2[i + 1] + x2[i + 2]) / (8 * T)

    x4 = x3 * x3

    x5 = np.zeros(x.shape)
    for i in range(N, len(x4) - N):
        for j in range(N):
            x5[i] += x4[i - j]
    x5 = x5 / N

    peaki = x5[0]
    spki = 0
    npki = 0
    c = 0
    peak = [0]
    threshold1 = spki
    pk = []
    for i in range(1, len(x5)):
        if x5[i] > peaki:
            peaki = x5[i]

        npki = ((npki * (i - 1)) + x5[i]) / i
        spki = ((spki * (i - 1)) + x5[i]) / i
        spki = 0.875 * spki + 0.125 * peaki
        npki = 0.875 * npki + 0.125 * peaki

        threshold1 = npki + 0.25 * (spki - npki)
        threshold2 = 0.5 * threshold1

        if (x5[i] >= threshold2):

            if (peak[-1] + N < i):
                peak.append(i)
                pk.append(x5[i])

    p = np.zeros(len(x5))
    rPeak = []
    Q = np.zeros(2)
    S = np.zeros(2)
    THR = 50
    for i in peak:
        if (i == 0 or i < 2 * rE):
            continue
        p[i] = 1

        ind = np.argmax(x2[i - rE:i + rE])
        maxIndexR = (ind + i - rE)
        rPeak.append(maxIndexR)
        # plt.plot(maxIndexR, x2[maxIndexR], 'ro', markersize=12)
        count_r.append(maxIndexR)
        prevDiffQ = 0
        prevDiffS = 0
        #    FIND THE Q POINT
        for i in range(1, THR):

            Q[0] = x2[maxIndexR - i]
            Q[1] = x2[maxIndexR - (i + 1)]

            diffQ = Q[0] - Q[1]

            if (diffQ < prevDiffQ):
                minIndexQ = maxIndexR - i
                break
            prevDiffQ = diffQ / 5

        # plt.plot(minIndexQ, x2[minIndexQ], 'bo', markersize=6)
        count_q.append(minIndexQ)

        #    FIND THE S POINT
        for i in range(1, THR):

            S[0] = x2[maxIndexR + i]
            S[1] = x2[maxIndexR + (i + 1)]

            diffS = S[0] - S[1]

            if (diffS < prevDiffS):
                minIndexS = maxIndexR + i
                break
            prevDiffS = diffS / 5

        # plt.plot(minIndexS, x2[minIndexS], 'go', markersize=6)
        count_s.append(minIndexS)
    rPeak = np.unique(rPeak)

    # plt.xlabel('time')
    # plt.show()
    return count_q, count_r, count_s


# 按12位取数据
def read_uint12(data_chunk):
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
    fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
    snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
    return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


# 取中段15秒数据
def getdata_10(data, sampling_rate):
    time_s = 15
    rdata = data
    signal_len = len(data)
    signal_len_10 = sampling_rate * time_s
    if signal_len < signal_len_10:
        pass
    else:
        offset = math.floor((signal_len - signal_len_10) / 2)
        rdata = data[offset:signal_len_10 + offset]
    return rdata


# 获取平均心率
def get_heart_rate_mean(ecg_data_path):
    # 单个信号位长度
    bit_len = 12
    # 采样率 320/s
    sampling_rate = 320


    f = open(ecg_data_path, 'rb')
    signal_bit = f.read()
    f.close()
    # 8位 to 12位
    signal = read_uint12(signal_bit)
    # data_dir = "../tset01/physionet.org/files/picsdb/1.0.0"
    # ecg_record = wfdb.rdsamp(f'{data_dir}/infant1_ecg')
    # signal=ecg_record[1][]
    # 分开3个通道
    signal_abc = np.array(signal).reshape(math.floor(len(signal) / 3), 3)
    # 2号通道取10秒
    signal_10 = getdata_10(signal_abc[:, 1], sampling_rate)
    # 拆分导联，滤肌电干扰、工频干扰
    f0_signal_b = filter_01(signal_10)
    # 滤基线漂移
    f1_signal_b = filter_02(f0_signal_b, sampling_rate)
    f1_signal_b = np.array(f1_signal_b)
    # 定位QRS
    count_q, count_r, count_s = getQRS(f1_signal_b)
    # 计算心率
    seconds = len(f1_signal_b) / sampling_rate
    heart_rate_mean = len(count_r) * 60 / seconds
    print(heart_rate_mean)
    # draw(f1_signal_b)
    return heart_rate_mean


##
if __name__ == '__main__':
    # ecg_data_path = "../tset01/physionet.org/files/picsdb/1.0.0"
    # ap = argparse.ArgumentParser()
    # #添加参数
    # ap.add_argument("-e", "--ecg", required=True, help="path to the input ecg data")
    # #参数解析
    # args = vars(ap.parse_args())
    # ecg_data_path = args['ecg']
    # print(ecg_data_path)
    get_heart_rate_mean("../tset01/physionet.org/files/picsdb/1.0.0/infant1_ecg")