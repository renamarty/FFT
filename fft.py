import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.io import wavfile as wav
import sounddevice as sd
import time

def prt(npa):
    print("Shape: ",npa.shape," size: ",npa.size)

rate, data = wav.read("guitar.wav")
# rate, data = wav.read("data/guitar.wav")
prt(data)

print(data)

# Shrink the input sound and play
data_sample= data[10000:50000]#10000:50000, 0]

sd.play(data_sample, rate)
time.sleep(3)

# do the FFT to Freq spectrum
fft_out = fft(data_sample)
fft_freq = fftfreq(data_sample.size,1/rate)
print( "FFT Shape ",fft_out.shape)

print("FFTSize: ", fft_out.size, " Freqsize ", fft_freq.size)
CUT= 25000
# fft_out[ CUT:(fft_out.size-CUT)] = 0

# yf[target_idx - 1 : target_idx + 2] = 0
# here is where we mess with the spectrum


# do the reverse FFT to get back the time spectrum
post_signal = ifft(fft_out)

# tamp down the junk
normalized_tone = np.int16((post_signal / (post_signal.max()*2)) * 32767)

# play normalized tone
sd.play(np.real(normalized_tone), rate)
sd.wait()

wav.write("data/filtered_guitar.wav", rate, np.abs(post_signal))
filename2 = 'data/filtered_guitar.wav'
#song2 = AudioSegment.from_wav(filename2)
#play(song2)

# matplotlib inline
fig, axs = plt.subplots(3)
# axs[0].subtitle("Raw audio signal")
axs[0].plot(data_sample)
# axs[1].subtitle("Frequency spectrum")
axs[1].plot(fft_freq,np.abs(fft_out))
# axs[1].plot(np.abs(fft_out))
axs[2].plot(normalized_tone)
plt.show()