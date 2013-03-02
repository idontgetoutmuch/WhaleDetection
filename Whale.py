import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.pylab as pl
import numpy as np
from matplotlib.mlab import specgram
import csv
import aifc
from sys import stdout
from local_settings import WHALE_HOME

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

long_specgram = lambda s: specgram(s, detrend=mpl.mlab.detrend_mean, NFFT=256, Fs=2, noverlap=178)
short_specgram = lambda s: specgram(s, detrend=mpl.mlab.detrend_mean, NFFT=128, Fs=2, noverlap=50)

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
                s = s/float(np.max(np.abs(s)))
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

def get_spectrogram(n, training=True):
    data = get_training_case(n, True, training)
    s,f,t = long_specgram(data)
    return s

def visualize_cases(cases=training_cases.keys()):
    if cases == "whales":
        cases = whale_cases
    elif cases == "no_whales":
        cases = no_whale_cases

    for n in cases:
        pl.clf()
        s = get_training_case(n);
        d_long,f_long,t_long=long_specgram(s)
        pl.subplot(211)
        pl
        pl.imshow(d_long*(d_long>0.7), aspect='auto')
        pl.subplot(212)
        pl.hist(d_long.flatten(), bins=100)
        if raw_input('q to stop; anything else to cont...') == 'q':
            break

def calculate_mean_over_class(cases=None, load_function=get_spectrogram):
    if cases is None:
        cases = training_cases.keys()
    elif cases == "whales":
        cases = whale_cases
    elif cases == "no_whales":
        cases = no_whale_cases

    N = len(cases)
    mean_data = load_function(cases[0])
    for ind, n in enumerate(cases[1:]):
        s = load_function(n)
        mean_data = mean_data + (s - mean_data)/(float (ind+2))

        if ind % 100 == 1:
            status_string = "\r["+"="*((ind*10)/N)+" "*(10-(ind*10)/N)+"] %d of %d"%(ind,N)
            stdout.write(status_string)
            stdout.flush()

    return mean_data

def translate_and_project_onto_vector(cases, t_vec, p_vec, load_function=get_spectrogram, training=True):
    t_vec = t_vec.flatten()
    p_vec = p_vec.flatten()
    mag = np.sqrt(np.dot(p_vec, p_vec))

    return [ np.dot(load_function(n, training).flatten() \
                        - t_vec, p_vec)/mag for n in cases]

