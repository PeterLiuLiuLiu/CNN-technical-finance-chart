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
import shutil
from sklearn.utils import shuffle
import re
import urllib.request as url
from mpl_finance import candlestick_ohlc

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',500)


def extract_aastock(path, threshold): # threshold in terms of 億
	df = pd.read_excel(path)
	for i in range(1, len(df)+1, 2):
	    df.iloc[i, 0], df.iloc[i-1, 0] = df.iloc[i-1, 0], df.iloc[i, 0]
	df.dropna(inplace = True)
	df = shuffle(df)
	for row in range(len(df)):
	    if df.iloc[row, 9][-1] == '億':
	        df.iloc[row, 9] = float(df.iloc[row, 9][: -1].replace(',',''))
	df = df[df.iloc[:, 9] > threshold]
	return [i[1:] for i in df.iloc[:,0]]


def extract_indexes(path):
	response = url.urlopen(path)
	text = str(response.read())
	indexes = re.search(r'"fallbackCategory":"YFINANCE:,*(.*?)","pageSubsite"', text).group(1)
	indexes = indexes.split(',')
	indexes.pop(indexes.index('^VIX'))
	indexes.pop(indexes.index('^HSI'))
	return shuffle(indexes)

HSI_CONPONENT_PATH = 'HSCE.xlsx'
TRAIN_VAL_SPLIT = 0.85
THRESHOLD = 100
NOS_TEST_STOCKS = 0

stock_list = extract_aastock(path = HSI_CONPONENT_PATH, threshold = THRESHOLD)
# stock_list = extract_indexes(r'https://sg.finance.yahoo.com/world-indices')
TRAIN_STOCKS = stock_list[:int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT)]
VAL_STOCKS = stock_list[int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT):]
# TEST_STOCKS = [stock_list[-NOS_TEST_STOCKS]] if NOS_TEST_STOCKS == 1 else stock_list[-NOS_TEST_STOCKS]
TEST_STOCKS = ['^HSCE', ]

# Printing interva will be the HCF of TIME_RANGE/INTERVAL_TO_PRINT and TIME_RANGE_2/INTERVAL_TO_PRINT
TIME_RANGE = 30
TIME_RANGE_2 = 90
INTERVAL_TO_PRINT = 2 # Must be a factor of TIME_RANGE & TIME_RANGE_2
PERIOD_TO_PREDICT = 5

BB_BAND = [20, 2]
band_type = ['up', 'down']
ATR_period = 14
RSI_period = 18
SMA = [BB_BAND[0], 100] 

FIGSIZE = (5, 3)
DPI = 400
PCT_BUY = 0.5 # Pct will be added

start_time = time.time()
date = datetime.datetime.utcfromtimestamp(int(start_time) + 8 * 3600).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 32
EPOCHS = 50
	
NAME = f'{date}--{TIME_RANGE}-SEQ-{PERIOD_TO_PREDICT}-PRED.h5'

up_path = os.path.join('Stock chart', 'up')
down_path = os.path.join('Stock chart', 'down')	
val_up_path = os.path.join('Val stock chart', 'up')
val_down_path = os.path.join('Val stock chart', 'down')	
test_up_path = os.path.join('Test stock chart', 'up')
test_down_path = os.path.join('Test stock chart', 'down')	


def change(current, current_close, stock_data):
	if current + PERIOD_TO_PREDICT < len(stock_data):
		if stock_data.loc[current + PERIOD_TO_PREDICT, 'Close'] < current_close * (1 + PCT_BUY / 100):
			return 0
		else:
			return 1


def plot_chart(up_path = up_path, down_path = down_path, \
			   return_num = False, STOCKS = TRAIN_STOCKS, TIME_RANGE = TIME_RANGE, \
				   INTERVAL_TO_PRINT = INTERVAL_TO_PRINT, REPLACE = False):
	if not return_num:
		
		if REPLACE:
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
			try:
				stock_data = web.DataReader(stock,'yahoo', start = '1970-01-30',end = datetime.datetime.today())
			except Exception:
				continue
			stock_data = stock_data.reset_index(drop = True)
			
			# SMA
			for sma in SMA:
				stock_data[f'{sma}_SMA'] = stock_data['Close'].rolling(sma).mean()
			
			# ATR
			stock_data['prev_cp'] = stock_data['Close'].shift(1)
			stock_data['ATR_H-L'] = stock_data['High'] - stock_data['Low']
			stock_data['ATR_H-L'] = stock_data['ATR_H-L'].abs()
			stock_data['ATR_H-Cp'] = stock_data['High'] - stock_data['prev_cp']
			stock_data['ATR_H-Cp'] = stock_data['ATR_H-Cp'].abs()
			stock_data['ATR_L-Cp'] = stock_data['Low'] - stock_data['prev_cp']
			stock_data['ATR_L-Cp'] = stock_data['ATR_L-Cp'].abs()
			stock_data['ATR'] = stock_data[['ATR_H-L', 'ATR_H-Cp','ATR_L-Cp']].max(axis=1)
			stock_data['ATR'] = stock_data['prev_cp'] / stock_data['ATR'].rolling(ATR_period).mean()
			stock_data.drop(columns = ['prev_cp', 'ATR_H-L', 'ATR_H-Cp', 'ATR_L-Cp'], inplace = True)
			
			# RSI
			stock_data['gain'] = stock_data.Close - stock_data.Close.shift(1)
			stock_data = stock_data.drop(columns = 'gain').join(stock_data.gain[stock_data.gain > 0])
			stock_data['loss'] = stock_data.Close - stock_data.Close.shift(1)
			stock_data = stock_data.drop(columns = 'loss').join(stock_data.loss[stock_data.loss < 0])
			stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].fillna(0)
			stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].rolling(RSI_period).mean()
			stock_data['RSI'] = stock_data.gain / stock_data['loss'].abs()
			stock_data['RSI'] = 100 - 100 / (1 + stock_data['RSI'])
			stock_data.drop(columns = ['gain', 'loss'], inplace = True)
			
			# BB
			for i, bb in enumerate(band_type, -1):
				mult = i if i < 0 else 1
				stock_data[f'BB{bb}'] = stock_data['Close'].rolling(BB_BAND[0]).mean() + \
					BB_BAND[1] * mult * stock_data['Close'].rolling(BB_BAND[0]).std()

			ohlc = deque(maxlen = TIME_RANGE)
			smas = deque(maxlen = TIME_RANGE)
			ohlc_2 = deque(maxlen = TIME_RANGE_2)
			smas_2 = deque(maxlen = TIME_RANGE_2)
			bband = deque(maxlen = TIME_RANGE_2)
			atr = deque(maxlen = TIME_RANGE_2)
			rsi = deque(maxlen = TIME_RANGE_2)
			refresh = 0
			
			for i, row in enumerate(stock_data.itertuples(), 1):
				if not row.Open == row.High:
					ohlc.append([row.Close])
					smas.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
					ohlc_2.append([i, row.Open, row.High, row.Low, row.Close])
					smas_2.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
					bband.append([row[np.where(stock_data.columns.values == f'BB{bb}')[0][0] + 1] for bb in band_type])
					atr.append(row.ATR)
					rsi.append(row.RSI)
					
				if len(ohlc_2) == TIME_RANGE_2 and i >= (max(SMA) + TIME_RANGE_2) and i >= ATR_period:
					if refresh % (TIME_RANGE/INTERVAL_TO_PRINT) == 0 or refresh % (TIME_RANGE_2/INTERVAL_TO_PRINT) == 0:
						fig = plt.figure(figsize = FIGSIZE)
						ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
# =============================================================================
# 						ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan = 1, sharex = ax1)
# # 						ax1.axis('off')
# # 						ax2.axis('off')
# 						print(i-TIME_RANGE_2, i)
# 						ax2.plot(range(i - TIME_RANGE_2, i), atr, linewidth = 3, color = 'blue')
# 						print(len(atr))
# 						ax2.plot(range(i - TIME_RANGE_2, i), rsi, linewidth = 3, color = 'red')
# 						ax1.fill_between(range(i - TIME_RANGE_2, i), [bband[a][0] for a in range(TIME_RANGE_2)], \
# 					   [bband[a][1] for a in range(TIME_RANGE_2)], facecolor = 'yellow')
# # 						for b in range(len(band_type)):
# #  							ax1.plot(range(TIME_RANGE_2), [bband[a][b] for a in range(TIME_RANGE_2)], color = 'blue', linewidth = 3)
# 						for b in range(len(SMA)):
#  							ax1.plot(range(i - TIME_RANGE_2, i), [smas_2[a][b] for a in range(TIME_RANGE_2)], color = 'red', linewidth = 2)
# =============================================================================
						candlestick_ohlc(ax1, ohlc_2)
# 						ax1.plot(range(i - TIME_RANGE, i), ohlc, linewidth = 2, color = 'black')
						time.sleep(3)
						up = change(i, row.Close, stock_data)
						if up == 1:
							filename = f'{stock}_{i}_{up}.png'
							filepath = os.path.join(up_path, filename)
							plt.xticks(range(i - TIME_RANGE_2, i, 5))
							plt.savefig(filepath, dpi = DPI)#, bbox_inches = 'tight', pad_inches = 0
							plt.close(fig)
						else:
							filename = f'{stock}_{i}_{up}.png'
							filepath = os.path.join(down_path, filename)
							plt.xticks(range(i - TIME_RANGE_2, i, 5))
							plt.savefig(filepath, dpi = DPI)#, bbox_inches = 'tight', pad_inches = 0
							plt.close(fig)
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
		

def plot_val_chart(val_up_path = val_up_path, val_down_path = val_down_path, \
					return_num = False, STOCKS = VAL_STOCKS, TIME_RANGE = TIME_RANGE, \
						INTERVAL_TO_PRINT = INTERVAL_TO_PRINT, REPLACE = False):
	if not return_num:	
		
		if REPLACE:
			try:
				shutil.rmtree(val_up_path)
				shutil.rmtree(val_down_path)
				os.makedirs(val_up_path)
				os.makedirs(val_down_path)
	
			except Exception:
				os.makedirs(val_up_path)
				os.makedirs(val_down_path)
		else:
			try:
				os.makedirs(val_up_path)
				os.makedirs(val_down_path)
			except Exception:
				pass
		
		print(f'Preparing validating data...')
		for stock in STOCKS:
			print(stock)
			try:
				stock_data = web.DataReader(stock,'yahoo', start = '1970-01-30',end = datetime.datetime.today())
			except Exception:
				continue
			stock_data = stock_data.reset_index(drop = True)
			
			# SMA
			for sma in SMA:
				stock_data[f'{sma}_SMA'] = stock_data['Close'].rolling(sma).mean()
			
			# ATR
			stock_data['prev_cp'] = stock_data['Close'].shift(1)
			stock_data['ATR_H-L'] = stock_data['High'] - stock_data['Low']
			stock_data['ATR_H-L'] = stock_data['ATR_H-L'].abs()
			stock_data['ATR_H-Cp'] = stock_data['High'] - stock_data['prev_cp']
			stock_data['ATR_H-Cp'] = stock_data['ATR_H-Cp'].abs()
			stock_data['ATR_L-Cp'] = stock_data['Low'] - stock_data['prev_cp']
			stock_data['ATR_L-Cp'] = stock_data['ATR_L-Cp'].abs()
			stock_data['ATR'] = stock_data[['ATR_H-L', 'ATR_H-Cp','ATR_L-Cp']].max(axis=1)
			stock_data['ATR'] = stock_data['prev_cp'] / stock_data['ATR'].rolling(ATR_period).mean()
			stock_data.drop(columns = ['prev_cp', 'ATR_H-L', 'ATR_H-Cp', 'ATR_L-Cp'], inplace = True)
			
			# RSI
			stock_data['gain'] = stock_data.Close - stock_data.Close.shift(1)
			stock_data = stock_data.drop(columns = 'gain').join(stock_data.gain[stock_data.gain > 0])
			stock_data['loss'] = stock_data.Close - stock_data.Close.shift(1)
			stock_data = stock_data.drop(columns = 'loss').join(stock_data.loss[stock_data.loss < 0])
			stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].fillna(0)
			stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].rolling(RSI_period).mean()
			stock_data['RSI'] = stock_data.gain / stock_data['loss'].abs()
			stock_data['RSI'] = 100 - 100 / (1 + stock_data['RSI'])
			stock_data.drop(columns = ['gain', 'loss'], inplace = True)
			
			# BB
			for i, bb in enumerate(band_type, -1):
				mult = i if i < 0 else 1
				stock_data[f'BB{bb}'] = stock_data['Close'].rolling(BB_BAND[0]).mean() + \
					BB_BAND[1] * mult * stock_data['Close'].rolling(BB_BAND[0]).std()

			ohlc = deque(maxlen = TIME_RANGE)
			smas = deque(maxlen = TIME_RANGE)
			ohlc_2 = deque(maxlen = TIME_RANGE_2)
			smas_2 = deque(maxlen = TIME_RANGE_2)
			bband = deque(maxlen = TIME_RANGE_2)
			atr = deque(maxlen = TIME_RANGE_2)
			rsi = deque(maxlen = TIME_RANGE_2)
			refresh = 0
			
			for i, row in enumerate(stock_data.itertuples(), 1):
				if not row.Open == row.High:
					ohlc.append([row.Close])
					smas.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
					ohlc_2.append([row.Close])
					smas_2.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
					bband.append([row[np.where(stock_data.columns.values == f'BB{bb}')[0][0] + 1] for bb in band_type])
					atr.append(row.ATR)
					rsi.append(row.RSI)
					
				if len(ohlc_2) == TIME_RANGE_2 and i >= (max(SMA) + TIME_RANGE_2) and i >= ATR_period:
					if refresh % (TIME_RANGE/INTERVAL_TO_PRINT) == 0 or refresh % (TIME_RANGE_2/INTERVAL_TO_PRINT) == 0:
						fig = plt.figure(figsize = FIGSIZE)
						ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
						ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan = 1)
						ax1.axis('off')
						ax2.axis('off')
						ax2.plot(range(TIME_RANGE_2), atr, linewidth = 3, color = 'blue')
						ax2.plot(range(TIME_RANGE_2), rsi, linewidth = 3, color = 'red')
						ax1.fill_between(range(TIME_RANGE_2), [bband[a][0] for a in range(TIME_RANGE_2)], \
					   [bband[a][1] for a in range(TIME_RANGE_2)], facecolor = 'yellow')
# 						for b in range(len(band_type)):
#  							ax1.plot(range(TIME_RANGE_2), [bband[a][b] for a in range(TIME_RANGE_2)], color = 'blue', linewidth = 3)
						for b in range(len(SMA)):
 							ax1.plot(range(TIME_RANGE_2), [smas_2[a][b] for a in range(TIME_RANGE_2)], color = 'red', linewidth = 2)
						ax1.plot(range(TIME_RANGE_2), ohlc_2, linewidth = 3, color = 'black')
						ax1.plot(range(TIME_RANGE_2 - TIME_RANGE, TIME_RANGE_2), ohlc, linewidth = 5, color = 'black')
						up = change(i, row.Close, stock_data)
						if up == 1:
							filename = f'{stock}_{i}_{up}.png'
							filepath = os.path.join(val_up_path, filename)
							plt.tight_layout()
							plt.savefig(filepath, dpi = DPI, bbox_inches = 'tight', pad_inches = 0)
							plt.close(fig)
						else:
							filename = f'{stock}_{i}_{up}.png'
							filepath = os.path.join(val_down_path, filename)
							plt.tight_layout()
							plt.savefig(filepath, dpi = DPI, bbox_inches = 'tight', pad_inches = 0)
							plt.close(fig)
					refresh += 1
						
		val_up_num, val_down_num = len(os.listdir(val_up_path)), len(os.listdir(val_down_path))
		remove_num = max(val_up_num, val_down_num) - min(val_up_num, val_down_num)
		if val_up_num > val_down_num:
			shuffled = os.listdir(val_up_path)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(val_up_path, shuffled[i]))
		else:
			shuffled = os.listdir(val_down_path)
			random.shuffle(shuffled)
			for i in range(remove_num):
				os.remove(os.path.join(val_down_path, shuffled[i]))
		print('\n')
				
	else:
		return len(os.listdir(val_up_path)), len(os.listdir(val_down_path))


def prepare_data():
	train_up_num, train_down_num = plot_chart(return_num = True)
	val_up_num, val_down_num = plot_val_chart(return_num = True)
	print(f'Nos. of train_data for ups and downs: {train_up_num}' if train_up_num == train_down_num \
		else f'Nos. of train_data for ups{(train_up_num)} and downs{(train_down_num)} are not equal')
	print(f'Nos. of val_data for ups and downs: {val_up_num}' if val_up_num == val_down_num \
		else f'Nos. of val_data for ups{(val_up_num)} and downs{(val_down_num)} are not equal')
	
	train, val = [], []
	train_X, train_y = [], []
	val_X, val_y = [], []
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
	for i, path in enumerate([val_down_path, val_up_path]):
		for file in os.listdir(path):
			img = cv2.imread(os.path.join(path, file), 0) / 255
			img = np.expand_dims(img, axis = 2)
			val.append([img, i])
	random.shuffle(val)
	val_X = [i[0] for i in val]
	val_y = [i[1] for i in val]		
	
	return np.array(train_X), np.array(train_y), \
		np.array(val_X), np.array(val_y)


plot_chart(REPLACE = True)
plot_val_chart(REPLACE = True)

train_X, train_y, val_X, val_y = prepare_data()

def Model_train_val_test():	
	
	model = models.Sequential()
	
	model.add(layers.Conv2D(64, (3, 3),
			  input_shape = (train_X.shape[1], train_X.shape[2], 1)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Dropout(0.4))
	
	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.45))
	
	model.add(layers.Conv2D(256, (3, 3)))
	model.add(layers.MaxPooling2D((2, 2)))	
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(0.5))
	
	model.add(layers.Flatten())
	model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)))
	model.add(layers.Activation('relu'))
	
	model.add(layers.Dense(1))
	model.add(layers.Activation('sigmoid'))
		  
	model.compile(loss='binary_crossentropy', 
				  optimizer = optimizers.RMSprop(lr=1e-4),
				  metrics=['accuracy'])
	
	history = model.fit(train_X, train_y, 
						batch_size = BATCH_SIZE,
						epochs = EPOCHS, 
						validation_data = (val_X, val_y))
	
	
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	
	try:
		os.mkdir('Models')
	except Exception:
		pass
	
	model.save(rf'Models\{NAME}')	
	
	plt.figure(figsize = (8, 6))
	plt.scatter(epochs, acc, label='Training acc', color = 'red', s = 5)
	plt.plot(epochs, val_acc, label='Validation acc')
	plt.title('Training and validation \n accuracy')
	plt.savefig(rf'Models\{date}__Train and val acc.png')
	
	plt.figure(figsize = (8, 6))
	plt.scatter(epochs, loss, label='Training loss', color = 'red', s = 5)
	plt.plot(epochs, val_loss, label='Validation loss')
	plt.title('Training and validation loss')
	plt.savefig(rf'Models\{date}__Train and val loss.png')


Model_train_val_test()


def Model_testing(test_up_path = test_up_path, test_down_path = test_down_path, \
			   return_num = False, STOCKS = TEST_STOCKS, TIME_RANGE = TIME_RANGE, \
				   INTERVAL_TO_PRINT = INTERVAL_TO_PRINT, REPLACE = True, \
					   RELOAD_MODEL = False):

	if REPLACE:
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
		
	print(f'Preparing and testing system...')
	for stock in STOCKS:
		print(stock)
		try:
			stock_data = web.DataReader(stock,'yahoo', start = '1970-01-30',end = datetime.datetime.today())
		except Exception:
			continue
		stock_data = stock_data.reset_index(drop = True)
		
		# SMA
		for sma in SMA:
			stock_data[f'{sma}_SMA'] = stock_data['Close'].rolling(sma).mean()
		
		# ATR
		stock_data['prev_cp'] = stock_data['Close'].shift(1)
		stock_data['ATR_H-L'] = stock_data['High'] - stock_data['Low']
		stock_data['ATR_H-L'] = stock_data['ATR_H-L'].abs()
		stock_data['ATR_H-Cp'] = stock_data['High'] - stock_data['prev_cp']
		stock_data['ATR_H-Cp'] = stock_data['ATR_H-Cp'].abs()
		stock_data['ATR_L-Cp'] = stock_data['Low'] - stock_data['prev_cp']
		stock_data['ATR_L-Cp'] = stock_data['ATR_L-Cp'].abs()
		stock_data['ATR'] = stock_data[['ATR_H-L', 'ATR_H-Cp','ATR_L-Cp']].max(axis=1)
		stock_data['ATR'] = stock_data['prev_cp'] / stock_data['ATR'].rolling(ATR_period).mean()
		stock_data.drop(columns = ['prev_cp', 'ATR_H-L', 'ATR_H-Cp', 'ATR_L-Cp'], inplace = True)
		
		# RSI
		stock_data['gain'] = stock_data.Close - stock_data.Close.shift(1)
		stock_data = stock_data.drop(columns = 'gain').join(stock_data.gain[stock_data.gain > 0])
		stock_data['loss'] = stock_data.Close - stock_data.Close.shift(1)
		stock_data = stock_data.drop(columns = 'loss').join(stock_data.loss[stock_data.loss < 0])
		stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].fillna(0)
		stock_data[['gain', 'loss']] = stock_data[['gain', 'loss']].rolling(RSI_period).mean()
		stock_data['RSI'] = stock_data.gain / stock_data['loss'].abs()
		stock_data['RSI'] = 100 - 100 / (1 + stock_data['RSI'])
		stock_data.drop(columns = ['gain', 'loss'], inplace = True)
		
		# BB
		for i, bb in enumerate(band_type, -1):
			mult = i if i < 0 else 1
			stock_data[f'BB{bb}'] = stock_data['Close'].rolling(BB_BAND[0]).mean() + \
				BB_BAND[1] * mult * stock_data['Close'].rolling(BB_BAND[0]).std()

		ohlc = deque(maxlen = TIME_RANGE)
		smas = deque(maxlen = TIME_RANGE)
		ohlc_2 = deque(maxlen = TIME_RANGE_2)
		smas_2 = deque(maxlen = TIME_RANGE_2)
		bband = deque(maxlen = TIME_RANGE_2)
		atr = deque(maxlen = TIME_RANGE_2)
		rsi = deque(maxlen = TIME_RANGE_2)
		refresh = 0
		
		for i, row in enumerate(stock_data.itertuples(), 1):
			if not row.Open == row.High:
				ohlc.append([row.Close])
				smas.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
				ohlc_2.append([row.Close])
				smas_2.append([row[np.where(stock_data.columns.values == f'{sma}_SMA')[0][0] + 1] for sma in SMA]) # To return the column position of smas
				bband.append([row[np.where(stock_data.columns.values == f'BB{bb}')[0][0] + 1] for bb in band_type])
				atr.append(row.ATR)
				rsi.append(row.RSI)
				
			if len(ohlc_2) == TIME_RANGE_2 and i >= (max(SMA) + TIME_RANGE_2) and i >= ATR_period:
				if refresh % (TIME_RANGE/INTERVAL_TO_PRINT) == 0 or refresh % (TIME_RANGE_2/INTERVAL_TO_PRINT) == 0:
					fig = plt.figure(figsize = FIGSIZE)
					ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
					ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan = 1)
					ax1.axis('off')
					ax2.axis('off')
					ax2.plot(range(TIME_RANGE_2), atr, linewidth = 3, color = 'blue')
					ax2.plot(range(TIME_RANGE_2), rsi, linewidth = 3, color = 'red')
					ax1.fill_between(range(TIME_RANGE_2), [bband[a][0] for a in range(TIME_RANGE_2)], \
				   [bband[a][1] for a in range(TIME_RANGE_2)], facecolor = 'yellow')
# 					for b in range(len(band_type)):
#  							ax1.plot(range(TIME_RANGE_2), [bband[a][b] for a in range(TIME_RANGE_2)], color = 'blue', linewidth = 3)
					for b in range(len(SMA)):
 							ax1.plot(range(TIME_RANGE_2), [smas_2[a][b] for a in range(TIME_RANGE_2)], color = 'red', linewidth = 2)
					ax1.plot(range(TIME_RANGE_2), ohlc_2, linewidth = 3, color = 'black')
					ax1.plot(range(TIME_RANGE_2 - TIME_RANGE, TIME_RANGE_2), ohlc, linewidth = 5, color = 'black')
					up = change(i, row.Close, stock_data)
					if up == 1:
						filename = f'{stock}_{i}_{up}.png'
						filepath = os.path.join(test_up_path, filename)
						plt.tight_layout()
						plt.savefig(filepath, dpi = DPI, bbox_inches = 'tight', pad_inches = 0)
						plt.close(fig)
					else:
						filename = f'{stock}_{i}_{up}.png'
						filepath = os.path.join(test_down_path, filename)
						plt.tight_layout()
						plt.savefig(filepath, dpi = DPI, bbox_inches = 'tight', pad_inches = 0)
						plt.close(fig)
				refresh += 1
					
	test_up_num, test_down_num = len(os.listdir(test_up_path)), len(os.listdir(test_down_path))
	print(f'Nos. of train_data for ups and downs: {test_up_num}' if test_up_num == test_down_num \
		else f'Nos. of train_data for ups: {(test_up_num)} and downs: {(test_down_num)}')
	
	test = []
	test_X, test_y = [], []
	for i, path in enumerate([test_down_path, test_up_path]):
		for file in os.listdir(path):
			img = cv2.imread(os.path.join(path, file), 0) / 255
			img = np.expand_dims(img, axis = 2)
			test.append([img, i])
	random.shuffle(test)
	test_X = [i[0] for i in test]
	test_y = [i[1] for i in test]

	if RELOAD_MODEL:
		TEST_MODEL_NAME = input('Please input the file name of the test model\n')
		model = load_model(rf'Models\{str(TEST_MODEL_NAME)}.h5')
	else:
		model = load_model(rf'Models\{NAME}')	
	
	model.summary()
	time.sleep(3)
	score = model.evaluate(np.array(test_X), np.array(test_y), verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


Model_testing(REPLACE = True, RELOAD_MODEL = False)






