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
	df = pd.read_csv('datamaps/datamap.csv')
	config = Config(22050, 0.02, 0.01, 13, 26, 512, 4)
	X_imf = [[], [], [], []]
	y_imf = [[], [], [], []]
	X_residue, y_residue = [], []
	pid_imf = [[], [], [], []]
	pid_residue = []

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
				pid_imf[i].append(pid)

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
			pid_residue.append(pid)
			print('write complete - ' + df.iloc[pid]['fname'])

	for i in range(config.max_imf):
		X, y, pid = np.array(X_imf[i]), np.array(y_imf[i]), np.array(pid_imf[i])
		pickle.dump(X, open('trainable/X_imf'+str(i)+'.pickle', 'wb'), protocol=4)
		pickle.dump(y, open('trainable/y_imf'+str(i)+'.pickle', 'wb'), protocol=4)
		pickle.dump(pid, open('pid/pid_imf'+str(i)+'.pickle', 'wb'), protocol=4)

	X, y, pid = np.array(X_residue), np.array(y_residue), np.array(pid_residue)
	pickle.dump(X, open('trainable/X_residue.pickle', 'wb'), protocol=4)
	pickle.dump(y, open('trainable/y_residue.pickle', 'wb'), protocol=4)
	pickle.dump(pid, open('pid/pid_residue.pickle', 'wb'), protocol=4)

	print('data has been written in /trainable')
