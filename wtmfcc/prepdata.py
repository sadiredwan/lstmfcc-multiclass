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
		self.wavelet = 'haar'
		self.mode = 'symmetric'
		self.order = 'freq'
		self.interpolation = 'nearest'
		self.maxlevel = 4


def rescale(signal, rate):
	n = len(signal)
	return np.interp(np.linspace(0, n, rate), np.arange(n), signal)

if __name__ == '__main__':
	df = pd.read_csv('datamaps/datamap.csv')
	config = Config(22050, 0.02, 0.01, 13, 26, 512)
	
	decomposed, mfcc_decomposed = [], []
	y_wt = []
	for i in range(len(df['fname'])):
		c = df.iloc[i]['class']
		f = df.iloc[i]['fname']
		signal, rate = librosa.load('data/'+c+'/'+f)
		signal = np.pad(signal, (0, rate-len(signal)), 'constant')
		
		wp = pywt.WaveletPacket(
			signal,
			wavelet=config.wavelet,
			mode=config.mode,
			maxlevel=config.maxlevel)
		
		values, mfcc_nodes = [], []
		for j in range(1, config.maxlevel+1):
			nodes = wp.get_level(j, order=config.order)
			labels = [n.path for n in nodes]
			value = np.array([rescale(n.data, rate) for n in nodes], 'd')
			
			mfcc_node = []
			for k, _ in enumerate(value):
				mfcc_node.append(mfcc(value[k],
						  samplerate=config.samplerate,
						  winlen=config.winlen,
						  winstep=config.winstep,
						  numcep=config.numcep,
						  nfilt=config.nfilt,
						  nfft=config.nfft))
			
# 			values.append(value)
			mfcc_nodes.append(mfcc_node)
		
# 		decomposed.append(values)
		mfcc_decomposed.append(mfcc_nodes)
		y_wt.append(df.iloc[i]['label'])
		print('write complete - ' + df.iloc[i]['fname'])
	
	X_wt = np.array(mfcc_decomposed).T.tolist()
	for i, X in enumerate(X_wt):
		X = np.array(X)	
		X_approx = []
		for feature in X:
			X_approx.append(feature[0])
		X_approx = np.array(X_approx)
		
		pickle.dump(X, open('trainable/X_wt_level'+str(i+1)+'.pickle', 'wb'), protocol=4)
		pickle.dump(X_approx, open('trainable/X_approx_level'+str(i+1)+'.pickle', 'wb'), protocol=4)
	
	pickle.dump(np.array(y_wt), open('trainable/y_wt.pickle', 'wb'), protocol=4)
	print('data has been written in /trainable')
