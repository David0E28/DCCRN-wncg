import os
import numpy as np
import random
import soundfile as sf
from numpy.linalg import norm


def signal_by_db(speech, noise, snr):
    speech = speech.astype(np.int16)
    noise = noise.astype(np.int16)

    len_speech = speech.shape[0]
    len_noise = noise.shape[0]
    start = random.randint(0, len_noise - len_speech)
    end = start + len_speech

    add_noise = noise[start:end]

    add_noise = add_noise / norm(add_noise) * norm(speech) / (10.0 ** (0.05 * snr))
    mix = speech + add_noise
    return mix


if __name__ == "__main__":

    noise_path = 'A:/WORK/project/VOICE_DATA/TIMIT/noise'
    # noises = ['babble', 'buccaneer1', 'destroy','factory1','volvo','white']
    noises = ['babble', ]
    clean_files = np.loadtxt('M:/DCCRN/scp/train.scp', dtype='str').tolist()
    path_noisy = 'A:/WORK/project/VOICE_DATA/TIMIT/noisy'

    snrs = [-5, 0, 5]

    with open('M:/DCCRN/scp/train_DNN_enh.scp', 'w+') as f:

        for noise in noises:
            print(noise)
            noise_file = os.path.join(noise_path, noise + '.wav')
            noise_data, fs = sf.read(noise_file, dtype='int16')

            for clean_file in clean_files:
                clean_path, clean_wav = os.path.split(clean_file)
                clean_data, fs = sf.read(clean_file, dtype='int16')

                for snr in snrs:
                    noisy_file = os.path.join(path_noisy, noise, str(snr), clean_wav)

                    noisy_path, _ = os.path.split(noisy_file)
                    os.makedirs(noisy_path, exist_ok=True)
                    mix = signal_by_db(clean_data, noise_data, snr)
                    noisy_data = np.asarray(mix, dtype=np.int16)
                    sf.write(noisy_file, noisy_data, fs)
                    f.write('%s %s\n' % (noisy_file, clean_file))
                    # print('%s %s\n'%(noisy_file,clean_file))
    f.close()
