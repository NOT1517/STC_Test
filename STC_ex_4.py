import numpy as np
import matplotlib.pyplot as plt

def spectral_density(freq, p_n, h_f, K, f, f_pom_n, f_pom_v, f_pom_s):
    if freq >= f_pom_n and freq < f_pom_s:
        return (h_f ** 2 * p_n ** 2 * 1e6 / K / f_pom_s)
    elif freq >= f_pom_s and freq <= f_pom_v:
        return (h_f ** 2 * p_n ** 2 * 1e6 / K / f ** 2)
    else:
        return 0

def generate_noise_signal(f, p_n, h_f, K, f_pom_n, f_pom_v, f_pom_s, fs):
    # np.random.seed(2)
    freq = np.linspace(0, fs / 2, fs // 2 + 1)

    noise_freq = [np.random.normal(0, np.sqrt(spectral_density(cur_freq, p_n, h_f, K, f, f_pom_n, f_pom_v, f_pom_s))) for cur_freq in freq]
    noise_time = np.fft.irfft(noise_freq)

    return noise_time, np.linspace(0, 1, fs, endpoint=False)

noise_TD, time = generate_noise_signal(f=1000,                      # частота, Гц
                                       p_n=0.1,                     # приведенное давление помех на входе, Па
                                       h_f=100e-6,                  # чувствительность гидрофона, В/Па
                                       K=2,                         # коэффициент помехоустойчивости
                                       f_pom_n=500,                 # нижнее значение частоты модели помехи, Гц
                                       f_pom_v=10000,               # верхнее значение частоты модели помехи, Гц
                                       f_pom_s=7500, fs=30000       # среднее значение частоты модели помехи, Гц
                                       )

# Отображение
plt.figure(figsize=(12, 6))
plt.plot(time, noise_TD, 'o-', markersize=2)
plt.title('Шумовой сигнал во времени')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда, В')
plt.grid()

# EXTRA GRAPHS
# plt.figure(figsize=(12, 6))
# plt.subplot(3, 1, 1)
# plt.magnitude_spectrum(noise_TD, Fs=30000)
# plt.title('Спектр шума')
# plt.xlabel('Частота, Гц')
# plt.ylabel('Амплитуда, В')
# plt.grid()
#
# plt.subplot(3, 1, 2)
# freq = np.linspace(0, 30000 / 2, 30000 // 2 + 1)
# plt.plot(freq, [spectral_density(cur_freq, f=1000, p_n=0.1, h_f=100e-6, K=2, f_pom_n=500, f_pom_v=10000, f_pom_s=7500) for cur_freq in freq])
# plt.title('Спектральная плотность шума')
# plt.xlabel('Частота, Гц')
# plt.ylabel('Мощность, Вт')
# plt.grid()
# #
# plt.subplot(3, 1, 3)
# plt.hist(np.fft.fft(noise_TD)[500:7500], 100, label='Fpom_n - Fpom_s')
# plt.hist(np.fft.fft(noise_TD)[7500:10000], 50, label='Fpom_s - Fpom_v')
# plt.title('Плотность вероятности')
# plt.xlabel('')
# plt.ylabel('')
# plt.grid()
# plt.legend()

plt.tight_layout()

plt.show()