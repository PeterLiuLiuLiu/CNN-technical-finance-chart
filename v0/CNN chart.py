# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:21:43 2019

@author: raymond-cy.liu
"""

import pandas as pd
import pandas_datareader.data as web
import datetime
import time
import os
import cv2
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import shutil
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

STOCKS = ['^DJI', '^GSPC', '^IXIC','^RUT', '^BFX', '^N225', '^HSI', \
		  '000001.SS', '^TWII', '^N100', '^NYA', '^XAX', 'IMOEX.ME', \
			  '^MXX', '^KS11', '^GDAXI']
TEST_STOCKS = ['^FCHI', '^JKSE']
TIME_RANGE = 60
INTERVAL_TO_PRINT = 5 # Must be a factor of TIME_RANGE
PERIOD_TO_PREDICT = 3

BATCH_SIZE = 10
EPOCHS = 40

up_path = os.path.join('Stock chart', 'up')
down_path = os.path.join('Stock chart', 'down')	
test_up_path = os.path.join('Test stock chart', 'up')
test_down_path = os.path.join('Test stock chart', 'down')	


def change(current, current_close, stock_data):
	if current + PERIOD_TO_PREDICT < len(stock_data):
		if stock_data.loc[current + PERIOD_TO_PREDICT, 'Close'] < current_close:
			return 0
		else:
			return 1


def plot_chart(up_path = up_path, down_path = down_path, \
			   return_num = False, STOCKS = STOCKS, TIME_RANGE = TIME_RANGE, \
				   INTERVAL_TO_PRINT = INTERVAL_TO_PRINT, replace = False):
	if not return_num:
		
		if replace:
			try:
				shutil.rmtree(up_path)
				shutil.rmtree(down_path)
				os.makedirs(up_path)
				os.makedirs(down_path)
			except Exception:
				os.makedirs(up_path)
				os.makedirs(down_path)
		else:
			try:
				os.makedirs(up_path)
				os.makedirs(down_path)
			except Exception:
				pass
			
		print(f'Preparing training data...')
		for stock in STOCKS:
			print(stock)
			stock_data = web.DataReader(stock,'yahoo', start = '1970-01-30',end = datetime.datetime.today())
			stock_data = stock_data.reset_index(drop=True)
			ohlc = deque(maxlen = TIME_RANGE)
			refresh = 0
			for i, row in enumerate(stock_data.itertuples(), 1):
				if not row.Open == row.High:
					ohlc.append([row.Index, row.Open, row.High, row.Low, row.Close, row.Volume])
					if len(ohlc) == TIME_RANGE:
						if refresh % TIME_RANGE/INTERVAL_TO_PRINT == 0:
							ax1 = plt.subplot2grid((1,1), (0,0))
							ax1.axis('off')
							candlestick_ohlc(ax1, ohlc, colorup='green', colordown = 'black')
							up = change(i, row.Close, stock_data)
							if up == 1:
								filename = f'{stock}_{i}_{up}.png'
								filepath = os.path.join(up_path, filename)
								plt.savefig(filepath, dpi = 100)
							else:
								filename = f'{stock}_{i}_{up}.png'
								filepath = os.path.join(down_path, filename)
								plt.savefig(filepath, dpi = 100)
						refresh += 1

		train_up_num, train_down_num = len(os.listdir(up_path)), len(os.listdir(down_path))
		remove_num = max(train_up_num, train_down_num) - min(train_up_num, train_down_num)
		if train_up_num > train_down_num:
			shuffled = os.listdir(up_path)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(up_path, shuffled[i]))
		else:
			shuffled = os.listdir(down_path)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(down_path, shuffled[i]))
		print('\n')
	else:
		return len(os.listdir(up_path)), len(os.listdir(down_path))
					
					
def plot_test_chart(test_up_path = test_up_path, test_down_path = test_down_path, \
					return_num = False, STOCKS = TEST_STOCKS, TIME_RANGE = TIME_RANGE, \
						INTERVAL_TO_PRINT = INTERVAL_TO_PRINT, replace = False):
	if not return_num:			
		if replace:
			try:
				shutil.rmtree(test_up_path)
				shutil.rmtree(test_down_path)
				os.makedirs(test_up_path)
				os.makedirs(test_down_path)
	
			except Exception:
				os.makedirs(test_up_path)
				os.makedirs(test_down_path)
		else:
			try:
				os.makedirs(test_up_path)
				os.makedirs(test_down_path)
			except Exception:
				pass
		
		print(f'Preparing testing data...')
		for stock in TEST_STOCKS:
			print(stock)
			stock_data = web.DataReader(stock,'yahoo', start = '1970-01-30',end = datetime.datetime.today())
			stock_data = stock_data.reset_index(drop=True)
			ohlc = deque(maxlen = TIME_RANGE)
			refresh = 0
			for i, row in enumerate(stock_data.itertuples(), 1):
				if not row.Open == row.High:
					ohlc.append([row.Index, row.Open, row.High, row.Low, row.Close, row.Volume])
					if len(ohlc) == TIME_RANGE:
						if refresh % TIME_RANGE/INTERVAL_TO_PRINT == 0:
							ax1 = plt.subplot2grid((1,1), (0,0))
							ax1.axis('off')
							candlestick_ohlc(ax1, ohlc, colorup='green', colordown = 'black')
							up = change(i, row.Close, stock_data)
							if up == 1:
								filename = f'{stock}_{i}_{up}.png'
								filepath = os.path.join(test_up_path, filename)
								plt.savefig(filepath, dpi = 100, figsize = (2,4))
							else:
								filename = f'{stock}_{i}_{up}.png'
								filepath = os.path.join(test_down_path, filename)
								plt.savefig(filepath, dpi = 100, figsize = (2, 4))
								plt.savefig()
						refresh += 1
		test_up_num, test_down_num = len(os.listdir(test_up_path)), len(os.listdir(test_down_path))
		remove_num = max(test_up_num, test_down_num) - min(test_up_num, test_down_num)
		if test_up_num > test_down_num:
			shuffled = os.listdir(test_up_path)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(test_up_path, shuffled[i]))
		else:
			shuffled = os.listdir(test_down_num)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(test_down_num, shuffled[i]))
		print('\n')
				
	else:
		return len(os.listdir(test_up_path)), len(os.listdir(test_down_path))


def prepare_data():
	train_up_num, train_down_num = plot_chart(return_num = True)
	test_up_num, test_down_num = plot_test_chart(return_num = True)
	print(f'Nos. of train_data for ups and downs: {train_up_num}' if train_up_num == train_down_num \
		else f'Nos. of train_data for ups{(train_up_num)} and downs{(train_down_num)} are not equal')
	print(f'Nos. of test_data for ups and downs: {test_up_num}' if test_up_num == test_down_num \
		else f'Nos. of test_data for ups{(test_up_num)} and downs{(test_down_num)} are not equal')
	
	train, test = [], []
	train_X, train_y = [], []
	test_X, test_y = [], []
	for i, path in enumerate([down_path, up_path]):
		for file in os.listdir(path):
			img = cv2.imread(os.path.join(path, file), 0) / 255
			img = np.expand_dims(img, axis = 2)
			train.append([img, i])
	random.shuffle(train)
	train_X = [i[0] for i in train]
	train_y = [i[1] for i in train]
	# 		cv2.namedWindow(file, cv2.WINDOW_NORMAL)
	# 		cv2.resize(img, IMG_SIZE)
	# 		cv2.imshow(file,img)
	# 		cv2.waitKey(0)
	# 		cv2.destroyAllWindows()
	for i, path in enumerate([test_down_path, test_up_path]):
		for file in os.listdir(path):
			img = cv2.imread(os.path.join(path, file), 0) / 255
			img = np.expand_dims(img, axis = 2)
			test.append([img, i])
	random.shuffle(test)
	test_X = [i[0] for i in test]
	test_y = [i[1] for i in test]		
	
	return np.array(train_X), np.array(train_y), \
		np.array(test_X), np.array(test_y)


# plot_chart(replace = True)
# plot_test_chart(replace = True)

train_X, train_y, test_X, test_y = prepare_data()

def Model_train_test():	
	
	model = models.Sequential()
	
	model.add(layers.Conv2D(32, (3, 3),
			  input_shape = (train_X.shape[1], train_X.shape[2], 1)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout(0.2))
	
	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.3))
	
	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.MaxPooling2D((2, 2)))	
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.4))
	
	model.add(layers.Flatten())
	model.add(layers.Dense(128))
	model.add(layers.Activation('relu'))
	
	model.add(layers.Dense(1))
	model.add(layers.Activation('sigmoid'))
		  
	model.compile(loss='binary_crossentropy', 
				  optimizer = optimizers.RMSprop(lr=1e-4),
				  metrics=['accuracy'])
	
	history = model.fit(train_X, train_y, 
						batch_size = BATCH_SIZE,
						epochs = EPOCHS, 
						validation_data = (test_X, test_y))
	
	
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	
	plt.scatter(epochs, acc, label='Training acc', color = 'red', s = 5)
	plt.plot(epochs, val_acc, label='Validation acc')
	plt.title('Training and validation \n accuracy')
	plt.figure(figsize = (8, 6))
	
	plt.scatter(epochs, loss, label='Training loss', color = 'red', s = 5)
	plt.plot(epochs, val_loss, label='Validation loss')
	plt.title('Training and validation loss')
	plt.figure(figsize = (8, 6))
	
	plt.show()

Model_train_test()




