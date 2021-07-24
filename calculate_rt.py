from scipy.signal import butter,filtfilt
import pandas as pd
import librosa
import numpy as np
from librosa import display

def main():
    input_file=input("Enter input file name, eg: sample.wav \n")
    output_file=input("Enter output file name with extension(.csv) to get it in Excel format, eg: sample.csv \n")
    response_time(input_file,output_file)

def response_time(input_file, output_file):
    
    def butter_highpass(data,cutoff, fs, order=5):
        """
       Design a highpass filter.
       Args:
       - cutoff (float) : the cutoff frequency of the filter.
       - fs     (float) : the sampling rate.
       - order    (int) : order of the filter, by default defined to 5.
       """
        nyq = 0.5 * fs
        high = cutoff / nyq
        b, a = butter(order, high, btype='high', analog=False)
        y = filtfilt(b, a, data)
        return y

    x, sr = librosa.load(input_file)
    x_f=butter_highpass(x,1000, sr, order=5)
    o_env = librosa.onset.onset_strength(x_f, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.util.peak_pick(o_env, 3, 1, 2, 3, 0.5, 50)

    D = np.abs(librosa.stft(x_f))
    
    onset_times = librosa.frames_to_time(onset_frames)
    a=onset_times.tolist()
    output=[]
    for i in range(0,len(a)-1, 2):
        output.append(a[i+1]-a[i])
    RT=pd.DataFrame(output, columns=['Duration in seconds'])
    RT['Trail']=[i for i in range(1,len(RT)+1)]
    RT[['Trail', 'Duration in seconds']].to_csv(output_file)
    print(f"The response time is successfully saved in the {output_file} file")


if __name__ == "__main__":
    main()