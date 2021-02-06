import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pickle
import numpy as np
from keras import Sequential
from keras import backend as K
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split
from kerastuner.engine.hyperparameters import HyperParameters 
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Flatten


def trial(hp):
	model = Sequential()
	model.add(TimeDistributed(Flatten(input_shape=X_train.shape[1:])))
	model.add(LSTM(hp.Int('lstm_1', min_value=32, max_value=256, step=32), return_sequences=True))
	model.add(LSTM(hp.Int('lstm_2', min_value=32, max_value=256, step=32), return_sequences=True))
	model.add(LSTM(hp.Int('lstm_3', min_value=32, max_value=256, step=32), return_sequences=True))
	model.add(LSTM(hp.Int('lstm_4', min_value=32, max_value=256, step=32), return_sequences=True))
	model.add(TimeDistributed(Dense(hp.Int('tdd_1', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_2', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_3', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(TimeDistributed(Dense(hp.Int('tdd_4', min_value=16, max_value=128, step=16), activation='relu')))
	model.add(Flatten())
	model.add(Dense(n_classes, activation='softmax'))
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer='adam',
		metrics=['acc'])
	return model


class RNN:
	def __init__(self, input_shape, output_shape, hyperparams):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.lstm_1 = hyperparams['lstm_1']
		self.lstm_2 = hyperparams['lstm_2']
		self.lstm_3 = hyperparams['lstm_3']
		self.lstm_4 = hyperparams['lstm_4']
		self.tdd_1 = hyperparams['tdd_1']
		self.tdd_2 = hyperparams['tdd_2']
		self.tdd_3 = hyperparams['tdd_3']
		self.tdd_4 = hyperparams['tdd_4']
	
	def run(self):
		model = Sequential()
		model.add(TimeDistributed(Flatten(input_shape=self.input_shape)))
		model.add(LSTM(self.lstm_1, return_sequences=True))
		model.add(LSTM(self.lstm_2, return_sequences=True))
		model.add(LSTM(self.lstm_3, return_sequences=True))
		model.add(LSTM(self.lstm_4, return_sequences=True))
		model.add(TimeDistributed(Dense(self.tdd_1, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_2, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_3, activation='relu')))
		model.add(TimeDistributed(Dense(self.tdd_4, activation='relu')))
		model.add(Flatten())
		model.add(Dense(self.output_shape, activation='softmax'))
		model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer='adam',
			metrics=['acc'])
		return model


def make_dataset():
	X, y = pickle.load(open('trainable/imfcc/X_all.pickle', 'rb')), pickle.load(open('trainable/imfcc/y_all.pickle', 'rb'))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
	X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2)
	pickle.dump(X_test, open('testdata/X_test_all.pickle', 'wb'))
	pickle.dump(y_test, open('testdata/y_test_all.pickle', 'wb'))

	return X_train, X_val, y_train, y_val


if __name__ == '__main__':

	X_train, X_val, y_train, y_val = make_dataset()
	n_classes = len(np.unique(y_train))
	
	LOG_DIR = 'log/'+f'{int(time.time())}'
	
	tuner = RandomSearch(
		trial,
		objective='val_acc',
		max_trials=1,
		executions_per_trial=1,
		directory=LOG_DIR)
	
	tuner.search(
		x=X_train,
		y=y_train,
		epochs=100,
		batch_size=50,
		shuffle='true',
		validation_data=(X_val, y_val))
	
	model = RNN(
		input_shape=X_train.shape[1:],
		output_shape = n_classes,
		hyperparams=tuner.get_best_hyperparameters()[0].values).run()

	hist = model.fit(
		X_train,
		y_train,
		epochs=100,
		batch_size=50,
		shuffle='true',
		validation_data=(X_val, y_val))

	model.summary()
	
	pickle.dump(hist.history, open('histories/imfcc_combined.pickle', 'wb'))
	model.save('models/model_imfcc_multiclass_winlen02.h5')
