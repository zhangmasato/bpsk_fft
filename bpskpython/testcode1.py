import numpy as np
import matplotlib.pyplot as plt

# 设置输入数据的位宽和长度
bit_width = 8
length = 8

# 生成随机的8位宽的二进制数字序列
# binary_sequence = np.random.randint(0, 2**bit_width, size=length)
# 生成全部为73的数字序列
binary_sequence = np.full(length, 73)


# BPSK调制器
def bpsk_modulator(decimal_data):
    if decimal_data < 0 or decimal_data > 255:
        raise ValueError("Decimal data should be in the range [0, 255].")

    binary_string = format(decimal_data, '08b')  # 将十进制数转换为8位二进制字符串
    encoded_data = []
    for bit in binary_string:
        if bit == '0':
            encoded_data.append(1)
        elif bit == '1':
            encoded_data.append(-1)
        else:
            raise ValueError("Invalid bit value. Only 0 or 1 allowed.")
    return encoded_data

# 对每个输入比特进行BPSK调制
bpsk_symbols = np.array([bpsk_modulator(data) for data in binary_sequence]).flatten()
print("BPSK Binary Sequence:")
print(bpsk_symbols)  

# 生成时间序列
time_sequence = np.arange(0, len(bpsk_symbols))
# IFFT模块
time_domain_signal = np.fft.ifft(bpsk_symbols)

# FFT变换
freq_domain_signal = np.fft.fft(time_domain_signal)
print("freq_domain_signal:")
print(freq_domain_signal)


# 绘制原始的二进制信号
plt.subplot(3, 1, 1)
plt.step(time_sequence, bpsk_symbols, where='post', label='Original BPSK Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('BPSK Modulated Waveform')
plt.ylim(-1.5, 1.5)  # 设置y轴范围
plt.grid(True)

# 绘制经过IFFT模块的频域信号
plt.subplot(3, 1, 2)
plt.step(time_sequence, np.real(time_domain_signal), where='post', label = 'real')
plt.step(time_sequence, np.imag(time_domain_signal), where='post', label = 'imag')
plt.title('IFFT Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# 绘制经过FFT模块的频域信号
plt.subplot(3, 1, 3)
plt.step(time_sequence, np.real(freq_domain_signal), where='post',label = 'real')
plt.step(time_sequence, np.imag(freq_domain_signal), where='post',label = 'imag')
plt.title('FFT Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()




plt.tight_layout()
plt.show()
