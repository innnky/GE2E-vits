import numpy as np

import GE2E
# print(GE2E.get_emb("/Volumes/Extend/AI/majo/sliced/ it's Motion.wav"))
if __name__ == '__main__':
    for line in open("filelists/train.txt").readlines():
        wav_path = line.split("|")[0]
        emb = GE2E.get_emb(wav_path)
        np.save(wav_path + ".spk.npy", emb)
