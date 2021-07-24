import librosa
from librosa import display
import matplotlib.pyplot as plt

def main():
    input_file=input("Enter input file name, eg: sample.wav \r")
    print("The sound wave will be visualised in few seconds...")
    y, sr = librosa.load(input_file)
    plt.figure(figsize=(20, 12))
    librosa.display.waveplot(y=y, sr=sr)
    plt.xlabel("time in sec")
    plt.ylabel("amplitude")
    plt.show()


if __name__ == "__main__":
    main()
