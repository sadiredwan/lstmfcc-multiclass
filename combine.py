import os
import pickle
import numpy as np


if __name__ == '__main__':
	pid = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
	for i, f_pid in enumerate(os.listdir(os.getcwd()+'/pid')):
		pid[i] = pickle.load(open('pid/'+f_pid, 'rb'))
	
	pid_all = pid[0]
	for p in pid:
		pid_all = np.intersect1d(pid_all, p)

	bad_imf = np.setdiff1d(pid[0], pid_all)
	bad_indices = [[], [], [], [], []]

	for i in range(len(pid)):
		bad_indices[i] = np.nonzero(np.in1d(pid[i], bad_imf))[0]

	X_all = []
	for i, f_X_imf in enumerate([x for x in os.listdir(os.getcwd()+'/trainable') if x.startswith('X')]):
		X_imf = pickle.load(open('trainable/'+f_X_imf, 'rb'))
		X_imf = np.delete(X_imf, bad_indices[i], 0)
		X_all.append(X_imf)

	#NOTE:
	#each mfcc feature vector for current configuration is of shape (99, 13)
	#each imf and the residue makes a dataset of shape (_testsize_, 99, 13)
	#the combined dataset with 4 imfs and the residue is of shape (_testsize_, 99, 5, 13)
	#however this yields input shape (99, 5, 13) for the LSTM units, valid input shape being (99, 13)
	#TODO:
	#use a Flatten layer wrapped in a TimeDistributed layer as the first layer for this dataset
	for i, _ in enumerate(X_all):
		X_all[i] = np.expand_dims(X_all[i], axis=2)

	X_all = np.concatenate(tuple(X_all), axis=2)

	y_all = pickle.load(open('trainable/y_imf0.pickle', 'rb'))
	y_all = np.delete(y_all, bad_indices[0], 0)

	pickle.dump(X_all, open('trainable/imfcc/X_all.pickle', 'wb'), protocol=4)
	pickle.dump(y_all, open('trainable/imfcc/y_all.pickle', 'wb'), protocol=4)
