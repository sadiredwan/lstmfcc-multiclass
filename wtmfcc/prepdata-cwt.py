import pywt
import pickle
import librosa
import numpy as np
import pandas as pd
from python_speech_features import mfcc


class Config:
	def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft):
		self.samplerate = samplerate
		self.winlen = winlen
		self.winstep = winstep
		self.numcep = numcep
		self.nfilt = nfilt
		self.nfft = nfft
		self.lowfreq = 0
		self.highfreq = None
		self.preemph = 0.97
		self.ceplifter = 22
		self.appendEnergy = True
		self.wavelet = 'morl'
		self.mode = 'symmetric'
		self.order = 'freq'
		self.interpolation = 'nearest'
		self.method = 'fft'
		self.maxlevel = 4
		self.scales = np.array([2**x for x in range(self.maxlevel)])


def rescale(signal, rate):
	n = len(signal)
	return np.interp(np.linspace(0, n, rate), np.arange(n), signal)

if __name__ == '__main__':
	df = pd.read_csv('datamaps/datamap.csv')
	config = Config(22050, 0.02, 0.01, 13, 26, 512)
	
	X_spec, y_spec = [], []
	X_coefs = [[], [], [], []]
	for i in range(len(df['fname'])):
		c = df.iloc[i]['class']
		f = df.iloc[i]['fname']
		signal, rate = librosa.load('data/'+c+'/'+f)
		signal = np.pad(signal, (0, rate-len(signal)), 'constant')
		
		coefs, freqs = pywt.cwt(
			data=signal,
			scales=config.scales,
			wavelet=config.wavelet,
			method=config.method)
		
		mfcc_coefs = []
		for j, _ in enumerate(coefs):
			mfcc_coefs.append(mfcc(coefs[j],
						   samplerate=config.samplerate,
						   winlen=config.winlen,
						   winstep=config.winstep,
						   numcep=config.numcep,
						   nfilt=config.nfilt,
						   nfft=config.nfft))
			X_coefs[j].append(mfcc_coefs[j])
		
		X_spec.append(mfcc_coefs)
		y_spec.append(df.iloc[i]['label'])
		print('write complete - ' + df.iloc[i]['fname'])

	for i, _ in enumerate(X_coefs):
		pickle.dump(np.array(X_coefs[i]), open('trainable-cwt/X_coefs'+str(i)+'.pickle', 'wb'), protocol=4)

	pickle.dump(np.array(X_spec), open('trainable-cwt/X_spec.pickle', 'wb'), protocol=4)
	pickle.dump(np.array(y_spec), open('trainable-cwt/y_spec.pickle', 'wb'), protocol=4)
	print('data has been written in /trainable-cwt')
	