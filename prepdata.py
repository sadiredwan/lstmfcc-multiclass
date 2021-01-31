import pickle
import librosa
import numpy as np
import pandas as pd
import concurrent.futures
from PyEMD import EMD
from python_speech_features import mfcc


class Config:
	def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft, max_imf):
		self.samplerate = samplerate
		self.winlen = winlen
		self.winstep = winstep
		self.numcep = numcep
		self.nfilt = nfilt
		self.nfft = nfft
		self.max_imf = max_imf
		self.lowfreq = 0
		self.highfreq = None
		self.preemph = 0.97
		self.ceplifter = 22
		self.appendEnergy = True


def get_imfs(pid, df, config):
	c = df.iloc[pid]['class']
	f = df.iloc[pid]['fname']
	signal, rate = librosa.load('data/'+c+'/'+f)
	emd = EMD()
	emd(signal, max_imf=config.max_imf)
	imfs, residue = emd.get_imfs_and_residue()
	return pid, imfs, residue, rate


if __name__ == '__main__':
	config = Config(22050, 0.02, 0.01, 13, 26, 512, 4)
	X_imf = [[], [], [], [], []]
	y_imf = [[], [], [], [], []]
	X_residue, y_residue = [], []
	df = pd.read_csv('datamaps/datamap.csv')

	with concurrent.futures.ProcessPoolExecutor() as executor:
		processes = []
		for i in range(len(df['fname'])):
			p = executor.submit(get_imfs, i, df, config)
			processes.append(p)

		for p in processes:
			pid, imfs, residue, rate = p.result()
			for i, imf in enumerate(imfs):
				imf = np.pad(imf, (0, rate-len(imf)), 'constant')
				imf = mfcc(imf,
					samplerate=config.samplerate,
					winlen=config.winlen,
					winstep=config.winstep,
					numcep=config.numcep,
					nfilt=config.nfilt,
					nfft=config.nfft)
				X_imf[i].append(imf)
				y_imf[i].append(df.iloc[pid]['label'])

			residue = np.pad(residue, (0, rate-len(residue)), 'constant')
			residue = mfcc(residue,
				samplerate=config.samplerate,
				winlen=config.winlen,
				winstep=config.winstep,
				numcep=config.numcep,
				nfilt=config.nfilt,
				nfft=config.nfft)
			X_residue.append(residue)
			y_residue.append(df.iloc[pid]['label'])
			print('write complete - ' + df.iloc[pid]['fname'])

	for i in range(config.max_imf):
		X, y = np.array(X_imf[i]), np.array(y_imf[i])
		X_out = open('trainable/X_imf'+str(i)+'.pickle', 'wb')
		pickle.dump(X, X_out)
		y_out = open('trainable/y_imf'+str(i)+'.pickle', 'wb')
		pickle.dump(y, y_out)
		X_out.close()
		y_out.close()

	X, y = np.array(X_residue), np.array(y_residue)
	X_out = open('trainable/X_residue.pickle', 'wb')
	pickle.dump(X, X_out)
	y_out = open('trainable/y_residue.pickle', 'wb')
	pickle.dump(y, y_out)
	X_out.close()
	y_out.close()
	print('data has been written in /trainable')
