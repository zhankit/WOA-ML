# # This is a sample Python script.
#
# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/



import wfdb
import matplotlib.pyplot as plt

# data_dir = "/physionet.org/files/picsdb/1.0.0"

# ecg_record = wfdb.rdsamp(f'{data_dir}/infant1_ecg/')
# plt.plot(ecg_record[0][:500])

ecg_record = wfdb.rdrecord('/physionet.org/files/picsdb/1.0.0/infant1_ecg/',
                           sampfrom=0,
                           sampto=10,
                           physical=False,
                           channels=[0,1])

signal = ecg_record.d_signal[0:1000]
plt.plot(signal)
plt.title("ECH Signal")
plt.show()