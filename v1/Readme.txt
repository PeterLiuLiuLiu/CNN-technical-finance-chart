1. 2020-01-02_1352--60-SEQ-3-PRED
CNN v1 aims to add three SMAs together with analyzing individual component stocks of certain indexes,setup as follws:

STOCKS = ['^DJI', '^GSPC', '^IXIC','^RUT', '^BFX', '^N225', '^HSI', \
		  '000001.SS', '^TWII', '^N100', '^NYA', '^XAX', 'IMOEX.ME', \
			  '^MXX', '^KS11', '^GDAXI']
VAL_STOCKS = ['^FCHI', '^JKSE']
TEST_STOCKS = ['^HSI']
TIME_RANGE = 60
INTERVAL_TO_PRINT = 5 # Must be a factor of TIME_RANGE
PERIOD_TO_PREDICT = 3
SMA = [15, 40, 60] #

start_time = time.time()
date = 	datetime.datetime.utcfromtimestamp(int(start_time)).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 10
EPOCHS = 30









2. 2020-01-03_1530--60-SEQ-3-PRED
Adding list of stocks from US S&P500 conpoenets, added the 2% margin for binary decision
TRAIN_STOCKS = list(pd.read_csv('constituents_csv.csv').Symbol)[:50]
VAL_STOCKS = ['^FCHI', '^DJI', '^GSPC']
TEST_STOCKS = ['^RUT']
TIME_RANGE = 60
INTERVAL_TO_PRINT = 5 # Must be a factor of TIME_RANGE
PERIOD_TO_PREDICT = 3
SMA = [20, 50, 100] #
FIGSIZE = (2,1)
PCT_BUY = 2 #Pct will be added

start_time = time.time()
date = 	datetime.datetime.utcfromtimestamp(int(start_time)).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 10
EPOCHS = 30
	
NAME = f'{date}--{TIME_RANGE}-SEQ-{PERIOD_TO_PREDICT}-PRED.h5'










3. Currency (gold) 1 min data, accuracy hovers around 50%
TRAIN_STOCKS = ['DAT_ASCII_XAUUSD_M1_2015.csv', 
				'DAT_ASCII_XAUUSD_M1_2016.csv',
				'DAT_ASCII_XAUUSD_M1_2017.csv']
VAL_STOCKS = ['DAT_ASCII_XAUUSD_M1_2018.csv', ]
TEST_STOCKS = ['DAT_ASCII_XAUUSD_M1_201901.csv',
			  'DAT_ASCII_XAUUSD_M1_201902.csv',
			  'DAT_ASCII_XAUUSD_M1_201903.csv',
			  'DAT_ASCII_XAUUSD_M1_201904.csv',
			  'DAT_ASCII_XAUUSD_M1_201905.csv',
			  'DAT_ASCII_XAUUSD_M1_201906.csv']
TIME_RANGE = 60
INTERVAL_TO_PRINT = 5 # Must be a factor of TIME_RANGE
PERIOD_TO_PREDICT = 5
SMA = [20, 50, 100] #
FIGSIZE = (2,1)
PCT_BUY = 0 #Pct will be added
RESAMPLE_TIME = '3Min'

Please input the file name of the test model
2020-01-06_0705--60-SEQ-5-PRED
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_12 (Conv2D)           (None, 98, 198, 32)       320       
_________________________________________________________________
activation_20 (Activation)   (None, 98, 198, 32)       0         
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 49, 99, 32)        0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 49, 99, 32)        0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 47, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 23, 48, 64)        0         
_________________________________________________________________
activation_21 (Activation)   (None, 23, 48, 64)        0         
_________________________________________________________________
dropout_13 (Dropout)         (None, 23, 48, 64)        0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 21, 46, 64)        36928     
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 10, 23, 64)        0         
_________________________________________________________________
activation_22 (Activation)   (None, 10, 23, 64)        0         
_________________________________________________________________
dropout_14 (Dropout)         (None, 10, 23, 64)        0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 14720)             0         
_________________________________________________________________
dense_8 (Dense)              (None, 128)               1884288   
_________________________________________________________________
activation_23 (Activation)   (None, 128)               0         
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_24 (Activation)   (None, 1)                 0         
=================================================================
Total params: 1,940,161
Trainable params: 1,940,161
Non-trainable params: 0
_________________________________________________________________
accuracy: 48.90%









4. Banking sector of HSI, got some degree of accuracy 

Preparing training data...
3618.HK
0023.HK
1288.HK
0011.HK
1111.HK
0939.HK
2388.HK
2066.HK
3698.HK
2888.HK
3988.HK
3328.HK
1658.HK
3866.HK
1988.HK
1398.HK
3968.HK


Preparing validating data...
0005.HK
2016.HK
0998.HK
2356.HK
6818.HK


Nos. of train_data for ups and downs: 2022
Nos. of val_data for ups and downs: 575
Train on 4044 samples, validate on 1150 samples
Epoch 1/40
4044/4044 [==============================] - 105s 26ms/sample - loss: 1.0770 - accuracy: 0.4990 - val_loss: 0.8675 - val_accuracy: 0.4939
Epoch 2/40
4044/4044 [==============================] - 113s 28ms/sample - loss: 0.7848 - accuracy: 0.4980 - val_loss: 0.7400 - val_accuracy: 0.4774
Epoch 3/40
4044/4044 [==============================] - 116s 29ms/sample - loss: 0.7283 - accuracy: 0.5059 - val_loss: 0.7176 - val_accuracy: 0.5104
Epoch 4/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.7120 - accuracy: 0.5059 - val_loss: 0.7075 - val_accuracy: 0.5000
Epoch 5/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.7054 - accuracy: 0.4973 - val_loss: 0.7031 - val_accuracy: 0.5087
Epoch 6/40
4044/4044 [==============================] - 106s 26ms/sample - loss: 0.7016 - accuracy: 0.5030 - val_loss: 0.7001 - val_accuracy: 0.5035
Epoch 7/40
4044/4044 [==============================] - 117s 29ms/sample - loss: 0.6989 - accuracy: 0.5141 - val_loss: 0.6981 - val_accuracy: 0.5096
Epoch 8/40
4044/4044 [==============================] - 110s 27ms/sample - loss: 0.6972 - accuracy: 0.5148 - val_loss: 0.6971 - val_accuracy: 0.5113
Epoch 9/40
4044/4044 [==============================] - 114s 28ms/sample - loss: 0.6960 - accuracy: 0.5282 - val_loss: 0.6966 - val_accuracy: 0.5165
Epoch 10/40
4044/4044 [==============================] - 110s 27ms/sample - loss: 0.6943 - accuracy: 0.5329 - val_loss: 0.6963 - val_accuracy: 0.5252
Epoch 11/40
4044/4044 [==============================] - 113s 28ms/sample - loss: 0.6935 - accuracy: 0.5396 - val_loss: 0.6962 - val_accuracy: 0.5017
Epoch 12/40
4044/4044 [==============================] - 110s 27ms/sample - loss: 0.6914 - accuracy: 0.5477 - val_loss: 0.6964 - val_accuracy: 0.4922
Epoch 13/40
4044/4044 [==============================] - 101s 25ms/sample - loss: 0.6902 - accuracy: 0.5509 - val_loss: 0.6969 - val_accuracy: 0.5357
Epoch 14/40
4044/4044 [==============================] - 101s 25ms/sample - loss: 0.6900 - accuracy: 0.5487 - val_loss: 0.6970 - val_accuracy: 0.5130
Epoch 15/40
4044/4044 [==============================] - 105s 26ms/sample - loss: 0.6892 - accuracy: 0.5608 - val_loss: 0.6967 - val_accuracy: 0.5243
Epoch 16/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.6887 - accuracy: 0.5591 - val_loss: 0.6971 - val_accuracy: 0.5183
Epoch 17/40
4044/4044 [==============================] - 109s 27ms/sample - loss: 0.6872 - accuracy: 0.5650 - val_loss: 0.6970 - val_accuracy: 0.5104
Epoch 18/40
4044/4044 [==============================] - 103s 26ms/sample - loss: 0.6863 - accuracy: 0.5631 - val_loss: 0.6967 - val_accuracy: 0.5200
Epoch 19/40
4044/4044 [==============================] - 103s 26ms/sample - loss: 0.6856 - accuracy: 0.5613 - val_loss: 0.6971 - val_accuracy: 0.5139
Epoch 20/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.6834 - accuracy: 0.5665 - val_loss: 0.6978 - val_accuracy: 0.5157
Epoch 21/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.6830 - accuracy: 0.5722 - val_loss: 0.6977 - val_accuracy: 0.5165
Epoch 22/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.6814 - accuracy: 0.5796 - val_loss: 0.6999 - val_accuracy: 0.5096
Epoch 23/40
4044/4044 [==============================] - 104s 26ms/sample - loss: 0.6813 - accuracy: 0.5762 - val_loss: 0.6983 - val_accuracy: 0.5235
Epoch 24/40
4044/4044 [==============================] - 106s 26ms/sample - loss: 0.6805 - accuracy: 0.5804 - val_loss: 0.6999 - val_accuracy: 0.5226
Epoch 25/40
4044/4044 [==============================] - 105s 26ms/sample - loss: 0.6760 - accuracy: 0.5977 - val_loss: 0.7011 - val_accuracy: 0.5235
Epoch 26/40
4044/4044 [==============================] - 106s 26ms/sample - loss: 0.6759 - accuracy: 0.5846 - val_loss: 0.7004 - val_accuracy: 0.5165
Epoch 27/40
4044/4044 [==============================] - 112s 28ms/sample - loss: 0.6761 - accuracy: 0.5848 - val_loss: 0.7028 - val_accuracy: 0.5209
Epoch 28/40
4044/4044 [==============================] - 107s 27ms/sample - loss: 0.6720 - accuracy: 0.6026 - val_loss: 0.7040 - val_accuracy: 0.5304
Epoch 29/40
4044/4044 [==============================] - 100s 25ms/sample - loss: 0.6722 - accuracy: 0.6026 - val_loss: 0.7037 - val_accuracy: 0.5339
Epoch 30/40
4044/4044 [==============================] - 101s 25ms/sample - loss: 0.6716 - accuracy: 0.5987 - val_loss: 0.7030 - val_accuracy: 0.5313
Epoch 31/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.6697 - accuracy: 0.6041 - val_loss: 0.7051 - val_accuracy: 0.5235
Epoch 32/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.6672 - accuracy: 0.6016 - val_loss: 0.7069 - val_accuracy: 0.5261
Epoch 33/40
4044/4044 [==============================] - 100s 25ms/sample - loss: 0.6648 - accuracy: 0.6130 - val_loss: 0.7073 - val_accuracy: 0.5209
Epoch 34/40
4044/4044 [==============================] - 107s 26ms/sample - loss: 0.6613 - accuracy: 0.6236 - val_loss: 0.7110 - val_accuracy: 0.5243
Epoch 35/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.6600 - accuracy: 0.6175 - val_loss: 0.7099 - val_accuracy: 0.5226
Epoch 36/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.6562 - accuracy: 0.6197 - val_loss: 0.7138 - val_accuracy: 0.5191
Epoch 37/40
4044/4044 [==============================] - 103s 26ms/sample - loss: 0.6546 - accuracy: 0.6301 - val_loss: 0.7146 - val_accuracy: 0.5261
Epoch 38/40
4044/4044 [==============================] - 103s 25ms/sample - loss: 0.6518 - accuracy: 0.6323 - val_loss: 0.7177 - val_accuracy: 0.5183
Epoch 39/40
4044/4044 [==============================] - 102s 25ms/sample - loss: 0.6443 - accuracy: 0.6355 - val_loss: 0.7217 - val_accuracy: 0.5287
Epoch 40/40
4044/4044 [==============================] - 105s 26ms/sample - loss: 0.6445 - accuracy: 0.6370 - val_loss: 0.7214 - val_accuracy: 0.5243
Preparing and testing system...
3968.HK
Nos. of train_data for ups: 121 and downs: 105
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 98, 198, 32)       320       
_________________________________________________________________
activation (Activation)      (None, 98, 198, 32)       0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 99, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 49, 99, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 23, 48, 64)        0         
_________________________________________________________________
activation_1 (Activation)    (None, 23, 48, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 23, 48, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 46, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 10, 23, 128)       0         
_________________________________________________________________
activation_2 (Activation)    (None, 10, 23, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 10, 23, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 29440)             0         
_________________________________________________________________
dense (Dense)                (None, 258)               7595778   
_________________________________________________________________
activation_3 (Activation)    (None, 258)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 259       
_________________________________________________________________
activation_4 (Activation)    (None, 1)                 0         
=================================================================
Total params: 7,688,709
Trainable params: 7,688,709
Non-trainable params: 0
_________________________________________________________________
accuracy: 55.31%



















5. HK red chip (^HSCC)
surprisingly the test result is not accurate despite of the 53.52% of val acc, seems like a mirror

runfile('C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v1/CNN chart v1_4 (for red chip).py', wdir='C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v1')
Preparing training data...
0688.HK
2319.HK
0981.HK
0371.HK
0144.HK
0152.HK
0165.HK
3320.HK
0941.HK
0257.HK
1199.HK
0270.HK
0135.HK
0392.HK
0267.HK
1109.HK
0883.HK
1193.HK
0836.HK
0817.HK
1313.HK


Preparing validating data...
0966.HK
0291.HK
1114.HK
0762.HK


Nos. of train_data for ups and downs: 3314
Nos. of val_data for ups and downs: 739
Train on 6628 samples, validate on 1478 samples
Epoch 1/60
WARNING:tensorflow:Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000001E991CD4158> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
WARNING: Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000001E991CD4158> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
6628/6628 [==============================] - 236s 36ms/sample - loss: 1.0070 - accuracy: 0.4887 - val_loss: 0.7885 - val_accuracy: 0.5014
Epoch 2/60
6628/6628 [==============================] - 234s 35ms/sample - loss: 0.7495 - accuracy: 0.4935 - val_loss: 0.7267 - val_accuracy: 0.5020
Epoch 3/60
6628/6628 [==============================] - 223s 34ms/sample - loss: 0.7162 - accuracy: 0.5077 - val_loss: 0.7078 - val_accuracy: 0.5088
Epoch 4/60
6628/6628 [==============================] - 221s 33ms/sample - loss: 0.7038 - accuracy: 0.4998 - val_loss: 0.7003 - val_accuracy: 0.5020
Epoch 5/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6985 - accuracy: 0.5006 - val_loss: 0.6963 - val_accuracy: 0.5000
Epoch 6/60
6628/6628 [==============================] - 220s 33ms/sample - loss: 0.6956 - accuracy: 0.4985 - val_loss: 0.6950 - val_accuracy: 0.5000
Epoch 7/60
6628/6628 [==============================] - 222s 33ms/sample - loss: 0.6946 - accuracy: 0.5054 - val_loss: 0.6943 - val_accuracy: 0.5000
Epoch 8/60
6628/6628 [==============================] - 230s 35ms/sample - loss: 0.6943 - accuracy: 0.4983 - val_loss: 0.6941 - val_accuracy: 0.4980
Epoch 9/60
6628/6628 [==============================] - 229s 35ms/sample - loss: 0.6941 - accuracy: 0.5063 - val_loss: 0.6941 - val_accuracy: 0.4919
Epoch 10/60
6628/6628 [==============================] - 223s 34ms/sample - loss: 0.6941 - accuracy: 0.5097 - val_loss: 0.6941 - val_accuracy: 0.5007
Epoch 11/60
6628/6628 [==============================] - 220s 33ms/sample - loss: 0.6940 - accuracy: 0.5041 - val_loss: 0.6940 - val_accuracy: 0.5000
Epoch 12/60
6628/6628 [==============================] - 221s 33ms/sample - loss: 0.6939 - accuracy: 0.5119 - val_loss: 0.6940 - val_accuracy: 0.5183
Epoch 13/60
6628/6628 [==============================] - 223s 34ms/sample - loss: 0.6937 - accuracy: 0.5174 - val_loss: 0.6940 - val_accuracy: 0.5047
Epoch 14/60
6628/6628 [==============================] - 219s 33ms/sample - loss: 0.6937 - accuracy: 0.5151 - val_loss: 0.6941 - val_accuracy: 0.5074
Epoch 15/60
6628/6628 [==============================] - 221s 33ms/sample - loss: 0.6930 - accuracy: 0.5256 - val_loss: 0.6942 - val_accuracy: 0.5122
Epoch 16/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6927 - accuracy: 0.5275 - val_loss: 0.6944 - val_accuracy: 0.5041
Epoch 17/60
6628/6628 [==============================] - 213s 32ms/sample - loss: 0.6927 - accuracy: 0.5315 - val_loss: 0.6945 - val_accuracy: 0.5041
Epoch 18/60
6628/6628 [==============================] - 220s 33ms/sample - loss: 0.6925 - accuracy: 0.5305 - val_loss: 0.6947 - val_accuracy: 0.5000
Epoch 19/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6919 - accuracy: 0.5300 - val_loss: 0.6950 - val_accuracy: 0.4973
Epoch 20/60
6628/6628 [==============================] - 215s 32ms/sample - loss: 0.6917 - accuracy: 0.5329 - val_loss: 0.6950 - val_accuracy: 0.4953
Epoch 21/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6914 - accuracy: 0.5352 - val_loss: 0.6954 - val_accuracy: 0.5061
Epoch 22/60
6628/6628 [==============================] - 215s 32ms/sample - loss: 0.6908 - accuracy: 0.5425 - val_loss: 0.6961 - val_accuracy: 0.5142
Epoch 23/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6898 - accuracy: 0.5489 - val_loss: 0.6961 - val_accuracy: 0.5196
Epoch 24/60
6628/6628 [==============================] - 215s 32ms/sample - loss: 0.6880 - accuracy: 0.5548 - val_loss: 0.6965 - val_accuracy: 0.5074
Epoch 25/60
6628/6628 [==============================] - 213s 32ms/sample - loss: 0.6879 - accuracy: 0.5501 - val_loss: 0.6968 - val_accuracy: 0.5122
Epoch 26/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6893 - accuracy: 0.5466 - val_loss: 0.6962 - val_accuracy: 0.5095
Epoch 27/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6876 - accuracy: 0.5674 - val_loss: 0.6974 - val_accuracy: 0.5095
Epoch 28/60
6628/6628 [==============================] - 228s 34ms/sample - loss: 0.6876 - accuracy: 0.5545 - val_loss: 0.6969 - val_accuracy: 0.5129
Epoch 29/60
6628/6628 [==============================] - 224s 34ms/sample - loss: 0.6864 - accuracy: 0.5594 - val_loss: 0.6975 - val_accuracy: 0.5217
Epoch 30/60
6628/6628 [==============================] - 221s 33ms/sample - loss: 0.6869 - accuracy: 0.5557 - val_loss: 0.6977 - val_accuracy: 0.5162
Epoch 31/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6864 - accuracy: 0.5596 - val_loss: 0.6974 - val_accuracy: 0.5237
Epoch 32/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6848 - accuracy: 0.5602 - val_loss: 0.6987 - val_accuracy: 0.5189
Epoch 33/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6838 - accuracy: 0.5647 - val_loss: 0.6991 - val_accuracy: 0.5223
Epoch 34/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6843 - accuracy: 0.5683 - val_loss: 0.6990 - val_accuracy: 0.5129
Epoch 35/60
6628/6628 [==============================] - 218s 33ms/sample - loss: 0.6837 - accuracy: 0.5661 - val_loss: 0.6990 - val_accuracy: 0.5027
Epoch 36/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6824 - accuracy: 0.5686 - val_loss: 0.6993 - val_accuracy: 0.5095
Epoch 37/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6819 - accuracy: 0.5732 - val_loss: 0.7005 - val_accuracy: 0.5244
Epoch 38/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6804 - accuracy: 0.5779 - val_loss: 0.6993 - val_accuracy: 0.5210
Epoch 39/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6807 - accuracy: 0.5762 - val_loss: 0.6982 - val_accuracy: 0.5162
Epoch 40/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6795 - accuracy: 0.5783 - val_loss: 0.6978 - val_accuracy: 0.5176
Epoch 41/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6767 - accuracy: 0.5833 - val_loss: 0.7012 - val_accuracy: 0.5203
Epoch 42/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6758 - accuracy: 0.5857 - val_loss: 0.7002 - val_accuracy: 0.5217
Epoch 43/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6749 - accuracy: 0.5845 - val_loss: 0.7033 - val_accuracy: 0.5142
Epoch 44/60
6628/6628 [==============================] - 215s 33ms/sample - loss: 0.6741 - accuracy: 0.5860 - val_loss: 0.7008 - val_accuracy: 0.5217
Epoch 45/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6715 - accuracy: 0.5973 - val_loss: 0.7021 - val_accuracy: 0.5298
Epoch 46/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6727 - accuracy: 0.5898 - val_loss: 0.7043 - val_accuracy: 0.5149
Epoch 47/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6699 - accuracy: 0.5991 - val_loss: 0.7041 - val_accuracy: 0.5264
Epoch 48/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6667 - accuracy: 0.5996 - val_loss: 0.7043 - val_accuracy: 0.5217
Epoch 49/60
6628/6628 [==============================] - 216s 33ms/sample - loss: 0.6664 - accuracy: 0.6040 - val_loss: 0.7036 - val_accuracy: 0.5277
Epoch 50/60
6628/6628 [==============================] - 213s 32ms/sample - loss: 0.6665 - accuracy: 0.6083 - val_loss: 0.7054 - val_accuracy: 0.5304
Epoch 51/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6617 - accuracy: 0.6079 - val_loss: 0.7069 - val_accuracy: 0.5311
Epoch 52/60
6628/6628 [==============================] - 215s 32ms/sample - loss: 0.6617 - accuracy: 0.6138 - val_loss: 0.7065 - val_accuracy: 0.5257
Epoch 53/60
6628/6628 [==============================] - 213s 32ms/sample - loss: 0.6557 - accuracy: 0.6166 - val_loss: 0.7089 - val_accuracy: 0.5203
Epoch 54/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6569 - accuracy: 0.6165 - val_loss: 0.7095 - val_accuracy: 0.5284
Epoch 55/60
6628/6628 [==============================] - 214s 32ms/sample - loss: 0.6525 - accuracy: 0.6187 - val_loss: 0.7135 - val_accuracy: 0.5298
Epoch 56/60
6628/6628 [==============================] - 219s 33ms/sample - loss: 0.6516 - accuracy: 0.6293 - val_loss: 0.7140 - val_accuracy: 0.5196
Epoch 57/60
6628/6628 [==============================] - 217s 33ms/sample - loss: 0.6491 - accuracy: 0.6293 - val_loss: 0.7163 - val_accuracy: 0.5244
Epoch 58/60
6628/6628 [==============================] - 213s 32ms/sample - loss: 0.6465 - accuracy: 0.6310 - val_loss: 0.7168 - val_accuracy: 0.5325
Epoch 59/60
6628/6628 [==============================] - 215s 32ms/sample - loss: 0.6415 - accuracy: 0.6364 - val_loss: 0.7159 - val_accuracy: 0.5257
Epoch 60/60
6628/6628 [==============================] - 212s 32ms/sample - loss: 0.6409 - accuracy: 0.6371 - val_loss: 0.7224 - val_accuracy: 0.5352



Preparing and testing system...
^HSCC
Nos. of train_data for ups: 76 and downs: 69

Please input the file name of the test model
2020-01-07_0145--60-SEQ-5-PRED
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 98, 198, 32)       320       
_________________________________________________________________
activation_5 (Activation)    (None, 98, 198, 32)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 49, 99, 32)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 49, 99, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 47, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 23, 48, 64)        0         
_________________________________________________________________
activation_6 (Activation)    (None, 23, 48, 64)        0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 23, 48, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 21, 46, 128)       73856     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 10, 23, 128)       0         
_________________________________________________________________
activation_7 (Activation)    (None, 10, 23, 128)       0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 10, 23, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 29440)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 258)               7595778   
_________________________________________________________________
activation_8 (Activation)    (None, 258)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 259       
_________________________________________________________________
activation_9 (Activation)    (None, 1)                 0         
=================================================================
Total params: 7,688,709
Trainable params: 7,688,709
Non-trainable params: 0
_________________________________________________________________
accuracy: 46.21%










6. HSI property sector
Nos. of train_data for ups and downs: 2627
Nos. of val_data for ups and downs: 555
Train on 5254 samples, validate on 1110 samples
Epoch 1/60
WARNING:tensorflow:Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000001E991C4C1E0> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
WARNING: Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000001E991C4C1E0> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
5254/5254 [==============================] - 167s 32ms/sample - loss: 1.0346 - accuracy: 0.5059 - val_loss: 0.8168 - val_accuracy: 0.5000
Epoch 2/60
5254/5254 [==============================] - 164s 31ms/sample - loss: 0.7611 - accuracy: 0.5040 - val_loss: 0.7299 - val_accuracy: 0.4982
Epoch 3/60
5254/5254 [==============================] - 163s 31ms/sample - loss: 0.7222 - accuracy: 0.4912 - val_loss: 0.7144 - val_accuracy: 0.4874
Epoch 4/60
5254/5254 [==============================] - 161s 31ms/sample - loss: 0.7105 - accuracy: 0.5065 - val_loss: 0.7070 - val_accuracy: 0.5000
Epoch 5/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.7044 - accuracy: 0.5036 - val_loss: 0.7019 - val_accuracy: 0.5000
Epoch 6/60
5254/5254 [==============================] - 157s 30ms/sample - loss: 0.7005 - accuracy: 0.5034 - val_loss: 0.6992 - val_accuracy: 0.5000
Epoch 7/60
5254/5254 [==============================] - 157s 30ms/sample - loss: 0.6985 - accuracy: 0.4966 - val_loss: 0.6972 - val_accuracy: 0.5000
Epoch 8/60
5254/5254 [==============================] - 160s 30ms/sample - loss: 0.6966 - accuracy: 0.5019 - val_loss: 0.6958 - val_accuracy: 0.4901
Epoch 9/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6957 - accuracy: 0.5021 - val_loss: 0.6951 - val_accuracy: 0.5090
Epoch 10/60
5254/5254 [==============================] - 157s 30ms/sample - loss: 0.6949 - accuracy: 0.5046 - val_loss: 0.6949 - val_accuracy: 0.5000
Epoch 11/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6949 - accuracy: 0.4954 - val_loss: 0.6945 - val_accuracy: 0.4928
Epoch 12/60
5254/5254 [==============================] - 161s 31ms/sample - loss: 0.6944 - accuracy: 0.5036 - val_loss: 0.6942 - val_accuracy: 0.5117
Epoch 13/60
5254/5254 [==============================] - 159s 30ms/sample - loss: 0.6943 - accuracy: 0.5032 - val_loss: 0.6941 - val_accuracy: 0.5144
Epoch 14/60
5254/5254 [==============================] - 162s 31ms/sample - loss: 0.6940 - accuracy: 0.5074 - val_loss: 0.6940 - val_accuracy: 0.4982
Epoch 15/60
5254/5254 [==============================] - 165s 31ms/sample - loss: 0.6938 - accuracy: 0.5118 - val_loss: 0.6940 - val_accuracy: 0.5099
Epoch 16/60
5254/5254 [==============================] - 163s 31ms/sample - loss: 0.6937 - accuracy: 0.5129 - val_loss: 0.6940 - val_accuracy: 0.5054
Epoch 17/60
5254/5254 [==============================] - 160s 31ms/sample - loss: 0.6931 - accuracy: 0.5253 - val_loss: 0.6940 - val_accuracy: 0.5063
Epoch 18/60
5254/5254 [==============================] - 162s 31ms/sample - loss: 0.6929 - accuracy: 0.5223 - val_loss: 0.6939 - val_accuracy: 0.5126
Epoch 19/60
5254/5254 [==============================] - 167s 32ms/sample - loss: 0.6928 - accuracy: 0.5226 - val_loss: 0.6940 - val_accuracy: 0.4982
Epoch 20/60
5254/5254 [==============================] - 163s 31ms/sample - loss: 0.6921 - accuracy: 0.5255 - val_loss: 0.6941 - val_accuracy: 0.5063
Epoch 21/60
5254/5254 [==============================] - 165s 31ms/sample - loss: 0.6921 - accuracy: 0.5320 - val_loss: 0.6943 - val_accuracy: 0.4982
Epoch 22/60
5254/5254 [==============================] - 162s 31ms/sample - loss: 0.6906 - accuracy: 0.5360 - val_loss: 0.6953 - val_accuracy: 0.5054
Epoch 23/60
5254/5254 [==============================] - 162s 31ms/sample - loss: 0.6909 - accuracy: 0.5343 - val_loss: 0.6943 - val_accuracy: 0.5027
Epoch 24/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6909 - accuracy: 0.5381 - val_loss: 0.6945 - val_accuracy: 0.5117
Epoch 25/60
5254/5254 [==============================] - 160s 30ms/sample - loss: 0.6897 - accuracy: 0.5394 - val_loss: 0.6935 - val_accuracy: 0.5243
Epoch 26/60
5254/5254 [==============================] - 161s 31ms/sample - loss: 0.6894 - accuracy: 0.5404 - val_loss: 0.6945 - val_accuracy: 0.5117
Epoch 27/60
5254/5254 [==============================] - 160s 30ms/sample - loss: 0.6890 - accuracy: 0.5440 - val_loss: 0.6941 - val_accuracy: 0.5180
Epoch 28/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6884 - accuracy: 0.5424 - val_loss: 0.6945 - val_accuracy: 0.5126
Epoch 29/60
5254/5254 [==============================] - 159s 30ms/sample - loss: 0.6878 - accuracy: 0.5451 - val_loss: 0.6934 - val_accuracy: 0.5189
Epoch 30/60
5254/5254 [==============================] - 161s 31ms/sample - loss: 0.6878 - accuracy: 0.5489 - val_loss: 0.6936 - val_accuracy: 0.5225
Epoch 31/60
5254/5254 [==============================] - 160s 30ms/sample - loss: 0.6853 - accuracy: 0.5561 - val_loss: 0.6957 - val_accuracy: 0.5198
Epoch 32/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6865 - accuracy: 0.5499 - val_loss: 0.6953 - val_accuracy: 0.5189
Epoch 33/60
5254/5254 [==============================] - 156s 30ms/sample - loss: 0.6857 - accuracy: 0.5600 - val_loss: 0.6954 - val_accuracy: 0.5144
Epoch 34/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6840 - accuracy: 0.5632 - val_loss: 0.6950 - val_accuracy: 0.5234
Epoch 35/60
5254/5254 [==============================] - 155s 29ms/sample - loss: 0.6843 - accuracy: 0.5579 - val_loss: 0.6949 - val_accuracy: 0.5279
Epoch 36/60
5254/5254 [==============================] - 160s 30ms/sample - loss: 0.6833 - accuracy: 0.5685 - val_loss: 0.6959 - val_accuracy: 0.5207
Epoch 37/60
5254/5254 [==============================] - 167s 32ms/sample - loss: 0.6820 - accuracy: 0.5662 - val_loss: 0.6979 - val_accuracy: 0.5171
Epoch 38/60
5254/5254 [==============================] - 168s 32ms/sample - loss: 0.6819 - accuracy: 0.5697 - val_loss: 0.6970 - val_accuracy: 0.5297
Epoch 39/60
5254/5254 [==============================] - 165s 31ms/sample - loss: 0.6814 - accuracy: 0.5655 - val_loss: 0.6992 - val_accuracy: 0.5297
Epoch 40/60
5254/5254 [==============================] - 157s 30ms/sample - loss: 0.6801 - accuracy: 0.5733 - val_loss: 0.6967 - val_accuracy: 0.5360
Epoch 41/60
5254/5254 [==============================] - 156s 30ms/sample - loss: 0.6808 - accuracy: 0.5737 - val_loss: 0.6983 - val_accuracy: 0.5333
Epoch 42/60
5254/5254 [==============================] - 160s 31ms/sample - loss: 0.6787 - accuracy: 0.5801 - val_loss: 0.6981 - val_accuracy: 0.5297
Epoch 43/60
5254/5254 [==============================] - 156s 30ms/sample - loss: 0.6775 - accuracy: 0.5794 - val_loss: 0.7006 - val_accuracy: 0.5315
Epoch 44/60
5254/5254 [==============================] - 156s 30ms/sample - loss: 0.6756 - accuracy: 0.5801 - val_loss: 0.6979 - val_accuracy: 0.5297
Epoch 45/60
5254/5254 [==============================] - 156s 30ms/sample - loss: 0.6750 - accuracy: 0.5780 - val_loss: 0.7009 - val_accuracy: 0.5252
Epoch 46/60
5254/5254 [==============================] - 158s 30ms/sample - loss: 0.6729 - accuracy: 0.5855 - val_loss: 0.7014 - val_accuracy: 0.5234
Epoch 47/60
5254/5254 [==============================] - 155s 29ms/sample - loss: 0.6714 - accuracy: 0.5919 - val_loss: 0.7066 - val_accuracy: 0.5135
Epoch 48/60
5254/5254 [==============================] - 157s 30ms/sample - loss: 0.6694 - accuracy: 0.6066 - val_loss: 0.7072 - val_accuracy: 0.5153
Epoch 49/60
5254/5254 [==============================] - 160s 31ms/sample - loss: 0.6667 - accuracy: 0.6049 - val_loss: 0.7014 - val_accuracy: 0.5243
Epoch 50/60
5254/5254 [==============================] - 186s 35ms/sample - loss: 0.6642 - accuracy: 0.6054 - val_loss: 0.7050 - val_accuracy: 0.5261
Epoch 51/60
5254/5254 [==============================] - 182s 35ms/sample - loss: 0.6623 - accuracy: 0.6079 - val_loss: 0.7053 - val_accuracy: 0.5324
Epoch 52/60
5254/5254 [==============================] - 182s 35ms/sample - loss: 0.6638 - accuracy: 0.6144 - val_loss: 0.7059 - val_accuracy: 0.5162
Epoch 53/60
5254/5254 [==============================] - 189s 36ms/sample - loss: 0.6603 - accuracy: 0.6152 - val_loss: 0.7120 - val_accuracy: 0.5252
Epoch 54/60
5254/5254 [==============================] - 185s 35ms/sample - loss: 0.6548 - accuracy: 0.6254 - val_loss: 0.7148 - val_accuracy: 0.5270
Epoch 55/60
5254/5254 [==============================] - 182s 35ms/sample - loss: 0.6515 - accuracy: 0.6292 - val_loss: 0.7106 - val_accuracy: 0.5216
Epoch 56/60
5254/5254 [==============================] - 198s 38ms/sample - loss: 0.6523 - accuracy: 0.6304 - val_loss: 0.7163 - val_accuracy: 0.5198
Epoch 57/60
5254/5254 [==============================] - 181s 34ms/sample - loss: 0.6484 - accuracy: 0.6268 - val_loss: 0.7146 - val_accuracy: 0.5144
Epoch 58/60
5254/5254 [==============================] - 190s 36ms/sample - loss: 0.6472 - accuracy: 0.6357 - val_loss: 0.7216 - val_accuracy: 0.5378
Epoch 59/60
5254/5254 [==============================] - 208s 40ms/sample - loss: 0.6435 - accuracy: 0.6403 - val_loss: 0.7227 - val_accuracy: 0.5135
Epoch 60/60
5254/5254 [==============================] - 193s 37ms/sample - loss: 0.6430 - accuracy: 0.6408 - val_loss: 0.7218 - val_accuracy: 0.5090

Preparing and testing system...
^HSNP
Nos. of train_data for ups: 154 and downs: 129

Please input the file name of the test model
2020-01-07_0906--30-SEQ-5-PRED
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 98, 198, 32)       320       
_________________________________________________________________
activation_10 (Activation)   (None, 98, 198, 32)       0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 49, 99, 32)        0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 49, 99, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 47, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 23, 48, 64)        0         
_________________________________________________________________
activation_11 (Activation)   (None, 23, 48, 64)        0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 23, 48, 64)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 21, 46, 128)       73856     
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 10, 23, 128)       0         
_________________________________________________________________
activation_12 (Activation)   (None, 10, 23, 128)       0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 10, 23, 128)       0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 29440)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 258)               7595778   
_________________________________________________________________
activation_13 (Activation)   (None, 258)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 259       
_________________________________________________________________
activation_14 (Activation)   (None, 1)                 0         
=================================================================
Total params: 7,688,709
Trainable params: 7,688,709
Non-trainable params: 0
_________________________________________________________________
accuracy: 52.65%















6. 