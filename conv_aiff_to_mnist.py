import matplotlib as mpl
from local_settings import WHALE_HOME
import csv
import aifc
import numpy as np
from matplotlib.mlab import specgram

def to_MNIST(n):
    spec_data = get_spectrogram(n)
    int_data = spec_data.astype(int)
    return int_data

long_specgram = lambda s: specgram(s, detrend=mpl.mlab.detrend_mean, NFFT=256, Fs=2, noverlap=178)

def get_spectrogram(n, training=True):
    data = get_training_case(n, True, training)
    s,f,t = long_specgram(data)
    s = 255.0 * s / np.max(np.abs(s))
    return s

def load_aiff(filename):
    snd = aifc.open(filename)
    snd_string = snd.readframes(snd.getnframes())
    snd.close()
    # need to do .byteswap as aifc loads / converts to a bytestring in
    # MSB ordering, but numpy assumes LSB ordering.
    return np.fromstring(snd_string, dtype=np.uint16).byteswap()

def get_training_case(n, normalised=True, training=True):
    if training:
        if n < 0 or n >= 30000:
            raise ValueError("training case out of range: %d" % n)
        else:
            filename = "%s/data/train/%s" % (WHALE_HOME, training_cases[n][0])
            s = load_aiff(filename)
            if normalised:
                s = s / float(np.max(np.abs(s)))
                return s
    else:
        if n < 1 or n > 54503:
            raise ValueError("submission case out of range: %d" % n)
        else:
            filename = "%s/data/test/test%s.aiff" % (WHALE_HOME, test_cases[n-1])
            s = load_aiff(filename)
            if normalised:
                s = s/float(np.max(np.abs(s)))
                return s

training_cases = {}
whale_cases = []
no_whale_cases = []

test_cases=[i for i in range(1,54504)]

with open("%s/data/train.csv"%WHALE_HOME, 'r') as csvfile:
    label_reader = csv.reader(csvfile)
    label_reader.next() # drop the headings
    for index, (file_name, label) in enumerate(label_reader):
        index = int(index)
        training_cases[index] = (file_name, 1 if label == '1' else 0)
        if label == '1':
            whale_cases.append(index)
        else:
            no_whale_cases.append(index)

for n in range(1, 30001):
    fn = 'data/train/MNIST%d.txt' % n
    with open(fn, "w") as output:
        mns = to_MNIST(n)
        np.savetxt(output, mns, fmt='%d')

