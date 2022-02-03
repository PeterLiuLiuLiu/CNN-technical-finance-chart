1. ####2020-01-07_1814--30-SEQ-5-PRED####
v2 first adds a subplot for a different timeframe (3 times) but with 4 nos. of SMAs, but its not useful to get the prediction score above half

HSI_CONPONENT_PATH = 'Banking sector HSI.xlsx'
TRAIN_VAL_SPLIT = 0.85
THRESHOLD = 100
NOS_TEST_STOCKS = 1

stock_list = extract_aastock(path = HSI_CONPONENT_PATH, threshold = THRESHOLD)
TRAIN_STOCKS = stock_list[:int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT)]
VAL_STOCKS = stock_list[int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT):]
TEST_STOCKS = [stock_list[-NOS_TEST_STOCKS]] if NOS_TEST_STOCKS == 1 else stock_list[-NOS_TEST_STOCKS]

# Printing interva will be the HCF of TIME_RANGE/INTERVAL_TO_PRINT and TIME_RANGE_2/INTERVAL_TO_PRINT
TIME_RANGE = 30
TIME_RANGE_2 = 90
INTERVAL_TO_PRINT = 3 # Must be a factor of TIME_RANGE & TIME_RANGE_2
PERIOD_TO_PREDICT = 5
SMA = [20, 50, 100, 250] 
FIGSIZE = (2,2)
DPI = 50
PCT_BUY = 0 # Pct will be added

start_time = time.time()
date = 	datetime.datetime.utcfromtimestamp(int(start_time) + 8 * 3600).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 32
EPOCHS = 30
	
NAME = f'{date}--{TIME_RANGE}-SEQ-{PERIOD_TO_PREDICT}-PRED.h5'

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








2. ####2020-01-07_2302--30-SEQ-5-PRED####

This model achieved ard 2% better than the normal (50%), with buy point set at 1%

HSI_CONPONENT_PATH = 'Banking sector HSI.xlsx'
TRAIN_VAL_SPLIT = 0.85
THRESHOLD = 100
NOS_TEST_STOCKS = 1

stock_list = extract_aastock(path = HSI_CONPONENT_PATH, threshold = THRESHOLD)
TRAIN_STOCKS = stock_list[:int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT)]
VAL_STOCKS = stock_list[int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT):]
TEST_STOCKS = [stock_list[-NOS_TEST_STOCKS]] if NOS_TEST_STOCKS == 1 else stock_list[-NOS_TEST_STOCKS]

# Printing interva will be the HCF of TIME_RANGE/INTERVAL_TO_PRINT and TIME_RANGE_2/INTERVAL_TO_PRINT
TIME_RANGE = 30
TIME_RANGE_2 = 60
INTERVAL_TO_PRINT = 2 # Must be a factor of TIME_RANGE & TIME_RANGE_2
PERIOD_TO_PREDICT = 5
SMA = [20, 50, 100, 250] 
FIGSIZE = (3, 1.5)
DPI = 60
PCT_BUY = 1 # Pct will be added

start_time = time.time()
date = 	datetime.datetime.utcfromtimestamp(int(start_time) + 8 * 3600).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 32
EPOCHS = 50
	
Preparing training data...
3618.HK
2356.HK
1111.HK
2388.HK
1288.HK
3988.HK
0023.HK
2066.HK
3698.HK
1988.HK
2888.HK
1398.HK
0005.HK
2016.HK
1658.HK
0011.HK
0939.HK


Preparing validating data...
0998.HK
6818.HK
3866.HK
3328.HK
3968.HK


Nos. of train_data for ups and downs: 1233
Nos. of val_data for ups and downs: 297
Preparing and testing system...
3968.HK
Nos. of train_data for ups: 85 and downs: 113

Please input the file name of the test model
dfd
Traceback (most recent call last):

  File "C:\Users\raymond-cy.liu\.spyder-py3\20191111 Building neutral nework\CNN chart\v2\CNN chart v2 (for HSI banking sector).py", line 476, in <module>
    Model_testing(REPLACE = True, RELOAD_MODEL = True)

  File "C:\Users\raymond-cy.liu\.spyder-py3\20191111 Building neutral nework\CNN chart\v2\CNN chart v2 (for HSI banking sector).py", line 466, in Model_testing
    model = load_model(rf'Models\{str(TEST_MODEL_NAME)}.h5')

  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow_core\python\keras\saving\save.py", line 149, in load_model
    loader_impl.parse_saved_model(filepath)

  File "C:\ProgramData\Anaconda3\lib\site-packages\tensorflow_core\python\saved_model\loader_impl.py", line 83, in parse_saved_model
    constants.SAVED_MODEL_FILENAME_PB))

OSError: SavedModel file does not exist at: Models\dfd.h5/{saved_model.pbtxt|saved_model.pb}


runfile('C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v2/CNN chart v2 (for HSI banking sector).py', wdir='C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v2')
Nos. of train_data for ups and downs: 1233
Nos. of val_data for ups and downs: 297
Train on 2466 samples, validate on 594 samples
Epoch 1/50
WARNING:tensorflow:Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000002D2D936C048> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
WARNING: Entity <function Function._initialize_uninitialized_variables.<locals>.initialize_variables at 0x000002D2D936C048> could not be transformed and will be executed as-is. Please report this to the AutoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: 
2466/2466 [==============================] - 132s 53ms/sample - loss: 1.6408 - accuracy: 0.4854 - val_loss: 1.2085 - val_accuracy: 0.5000
Epoch 2/50
2466/2466 [==============================] - 133s 54ms/sample - loss: 1.0035 - accuracy: 0.5028 - val_loss: 0.8521 - val_accuracy: 0.4832
Epoch 3/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.8048 - accuracy: 0.4939 - val_loss: 0.7681 - val_accuracy: 0.5000
Epoch 4/50
2466/2466 [==============================] - 126s 51ms/sample - loss: 0.7507 - accuracy: 0.5041 - val_loss: 0.7347 - val_accuracy: 0.5051
Epoch 5/50
2466/2466 [==============================] - 127s 51ms/sample - loss: 0.7257 - accuracy: 0.4955 - val_loss: 0.7170 - val_accuracy: 0.5017
Epoch 6/50
2466/2466 [==============================] - 131s 53ms/sample - loss: 0.7125 - accuracy: 0.5032 - val_loss: 0.7069 - val_accuracy: 0.5000
Epoch 7/50
2466/2466 [==============================] - 139s 56ms/sample - loss: 0.7040 - accuracy: 0.5008 - val_loss: 0.7009 - val_accuracy: 0.5000
Epoch 8/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6992 - accuracy: 0.5122 - val_loss: 0.6977 - val_accuracy: 0.4966
Epoch 9/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6973 - accuracy: 0.4988 - val_loss: 0.6962 - val_accuracy: 0.5000
Epoch 10/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6958 - accuracy: 0.4992 - val_loss: 0.6952 - val_accuracy: 0.5000
Epoch 11/50
2466/2466 [==============================] - 130s 53ms/sample - loss: 0.6949 - accuracy: 0.5081 - val_loss: 0.6946 - val_accuracy: 0.5000
Epoch 12/50
2466/2466 [==============================] - 137s 56ms/sample - loss: 0.6945 - accuracy: 0.4907 - val_loss: 0.6941 - val_accuracy: 0.5000
Epoch 13/50
2466/2466 [==============================] - 144s 58ms/sample - loss: 0.6943 - accuracy: 0.4935 - val_loss: 0.6940 - val_accuracy: 0.5000
Epoch 14/50
2466/2466 [==============================] - 133s 54ms/sample - loss: 0.6941 - accuracy: 0.4959 - val_loss: 0.6938 - val_accuracy: 0.5000
Epoch 15/50
2466/2466 [==============================] - 134s 54ms/sample - loss: 0.6938 - accuracy: 0.5020 - val_loss: 0.6936 - val_accuracy: 0.5000
Epoch 16/50
2466/2466 [==============================] - 137s 55ms/sample - loss: 0.6936 - accuracy: 0.4915 - val_loss: 0.6935 - val_accuracy: 0.5000
Epoch 17/50
2466/2466 [==============================] - 132s 54ms/sample - loss: 0.6936 - accuracy: 0.5008 - val_loss: 0.6934 - val_accuracy: 0.5000
Epoch 18/50
2466/2466 [==============================] - 130s 53ms/sample - loss: 0.6934 - accuracy: 0.4984 - val_loss: 0.6933 - val_accuracy: 0.5000
Epoch 19/50
2466/2466 [==============================] - 125s 51ms/sample - loss: 0.6933 - accuracy: 0.4935 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 20/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6933 - accuracy: 0.4911 - val_loss: 0.6933 - val_accuracy: 0.5000
Epoch 21/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6933 - accuracy: 0.4972 - val_loss: 0.6933 - val_accuracy: 0.5000
Epoch 22/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6933 - accuracy: 0.4980 - val_loss: 0.6933 - val_accuracy: 0.5000
Epoch 23/50
2466/2466 [==============================] - 125s 51ms/sample - loss: 0.6933 - accuracy: 0.4988 - val_loss: 0.6933 - val_accuracy: 0.5000
Epoch 24/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6933 - accuracy: 0.4874 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 25/50
2466/2466 [==============================] - 126s 51ms/sample - loss: 0.6933 - accuracy: 0.5036 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 26/50
2466/2466 [==============================] - 127s 51ms/sample - loss: 0.6932 - accuracy: 0.5065 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 27/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6933 - accuracy: 0.5024 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 28/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6934 - accuracy: 0.4968 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 29/50
2466/2466 [==============================] - 125s 51ms/sample - loss: 0.6933 - accuracy: 0.4996 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 30/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6933 - accuracy: 0.4931 - val_loss: 0.6932 - val_accuracy: 0.5051
Epoch 31/50
2466/2466 [==============================] - 127s 51ms/sample - loss: 0.6932 - accuracy: 0.4878 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 32/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6932 - accuracy: 0.4992 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 33/50
2466/2466 [==============================] - 126s 51ms/sample - loss: 0.6932 - accuracy: 0.5109 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 34/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6932 - accuracy: 0.5109 - val_loss: 0.6932 - val_accuracy: 0.5185
Epoch 35/50
2466/2466 [==============================] - 131s 53ms/sample - loss: 0.6932 - accuracy: 0.4976 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 36/50
2466/2466 [==============================] - 130s 53ms/sample - loss: 0.6932 - accuracy: 0.4959 - val_loss: 0.6932 - val_accuracy: 0.5000
Epoch 37/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6932 - accuracy: 0.5089 - val_loss: 0.6932 - val_accuracy: 0.5152
Epoch 38/50
2466/2466 [==============================] - 125s 51ms/sample - loss: 0.6932 - accuracy: 0.4968 - val_loss: 0.6932 - val_accuracy: 0.5253
Epoch 39/50
2466/2466 [==============================] - 126s 51ms/sample - loss: 0.6931 - accuracy: 0.5126 - val_loss: 0.6932 - val_accuracy: 0.5236
Epoch 40/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6931 - accuracy: 0.5101 - val_loss: 0.6932 - val_accuracy: 0.5236
Epoch 41/50
2466/2466 [==============================] - 129s 53ms/sample - loss: 0.6930 - accuracy: 0.5264 - val_loss: 0.6931 - val_accuracy: 0.5152
Epoch 42/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6928 - accuracy: 0.5211 - val_loss: 0.6931 - val_accuracy: 0.5185
Epoch 43/50
2466/2466 [==============================] - 125s 51ms/sample - loss: 0.6927 - accuracy: 0.5207 - val_loss: 0.6931 - val_accuracy: 0.5135
Epoch 44/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6924 - accuracy: 0.5260 - val_loss: 0.6930 - val_accuracy: 0.5135
Epoch 45/50
2466/2466 [==============================] - 129s 52ms/sample - loss: 0.6922 - accuracy: 0.5316 - val_loss: 0.6931 - val_accuracy: 0.5051
Epoch 46/50
2466/2466 [==============================] - 124s 50ms/sample - loss: 0.6920 - accuracy: 0.5251 - val_loss: 0.6929 - val_accuracy: 0.5168
Epoch 47/50
2466/2466 [==============================] - 128s 52ms/sample - loss: 0.6910 - accuracy: 0.5361 - val_loss: 0.6933 - val_accuracy: 0.5135
Epoch 48/50
2466/2466 [==============================] - 127s 52ms/sample - loss: 0.6911 - accuracy: 0.5247 - val_loss: 0.6933 - val_accuracy: 0.5051
Epoch 49/50
2466/2466 [==============================] - 130s 53ms/sample - loss: 0.6895 - accuracy: 0.5357 - val_loss: 0.6936 - val_accuracy: 0.5135
Epoch 50/50
2466/2466 [==============================] - 126s 51ms/sample - loss: 0.6900 - accuracy: 0.5349 - val_loss: 0.6938 - val_accuracy: 0.5101
Preparing and testing system...
6818.HK
Nos. of train_data for ups: 31 and downs: 48

Please input the file name of the test model
2020-01-07_2302--30-SEQ-5-PRED
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 88, 178, 64)       640       
_________________________________________________________________
activation_5 (Activation)    (None, 88, 178, 64)       0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 44, 89, 64)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 44, 89, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 42, 87, 128)       73856     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 21, 43, 128)       0         
_________________________________________________________________
activation_6 (Activation)    (None, 21, 43, 128)       0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 21, 43, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 19, 41, 256)       295168    
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 9, 20, 256)        0         
_________________________________________________________________
activation_7 (Activation)    (None, 9, 20, 256)        0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 9, 20, 256)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 46080)             0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               23593472  
_________________________________________________________________
activation_8 (Activation)    (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
_________________________________________________________________
activation_9 (Activation)    (None, 1)                 0         
=================================================================
Total params: 23,963,649
Trainable params: 23,963,649
Non-trainable params: 0
_________________________________________________________________
accuracy: 51.90%


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



3. ####2020-01-08_1420--30-SEQ-5-PRED####
The inclusion of BB band didnt help much, the val accuracy isnt as high but the testing is higher

HSI_CONPONENT_PATH = 'Property developer sector.xlsx'
TRAIN_VAL_SPLIT = 0.85
THRESHOLD = 100
NOS_TEST_STOCKS = 1

stock_list = extract_aastock(path = HSI_CONPONENT_PATH, threshold = THRESHOLD)
TRAIN_STOCKS = stock_list[:int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT)]
VAL_STOCKS = stock_list[int((len(stock_list) - NOS_TEST_STOCKS) * TRAIN_VAL_SPLIT):]
TEST_STOCKS = [stock_list[-NOS_TEST_STOCKS]] if NOS_TEST_STOCKS == 1 else stock_list[-NOS_TEST_STOCKS]

# Printing interva will be the HCF of TIME_RANGE/INTERVAL_TO_PRINT and TIME_RANGE_2/INTERVAL_TO_PRINT
TIME_RANGE = 30
TIME_RANGE_2 = 60
INTERVAL_TO_PRINT = 2 # Must be a factor of TIME_RANGE & TIME_RANGE_2
PERIOD_TO_PREDICT = 5
BB_BAND = [20, 2]
SMA = [BB_BAND[0], 50, 125, 250] 
band_type = ['up', 'down']
FIGSIZE = (3, 1.5)
DPI = 60
PCT_BUY = 0.5 # Pct will be added

start_time = time.time()
date = datetime.datetime.utcfromtimestamp(int(start_time) + 8 * 3600).strftime("%Y-%m-%d_%H%M")
BATCH_SIZE = 32
EPOCHS = 50

runfile('C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v2/CNN chart v2.py', wdir='C:/Users/raymond-cy.liu/.spyder-py3/20191111 Building neutral nework/CNN chart/v2')
Preparing training data...
0012.HK
2007.HK
0688.HK
1997.HK
0101.HK
0083.HK
1109.HK
0823.HK


Preparing validating data...
0017.HK
1113.HK
0016.HK


Nos. of train_data for ups and downs: 915
Nos. of val_data for ups and downs: 325
Train on 1830 samples, validate on 650 samples
Epoch 1/50
1830/1830 [==============================] - 101s 55ms/sample - loss: 1.6849 - accuracy: 0.5044 - val_loss: 1.3334 - val_accuracy: 0.5000
Epoch 2/50
1830/1830 [==============================] - 98s 54ms/sample - loss: 1.1418 - accuracy: 0.4852 - val_loss: 0.9565 - val_accuracy: 0.5031
Epoch 3/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.8706 - accuracy: 0.4973 - val_loss: 0.8060 - val_accuracy: 0.5000
Epoch 4/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.7802 - accuracy: 0.4913 - val_loss: 0.7574 - val_accuracy: 0.5000
Epoch 5/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.7462 - accuracy: 0.5087 - val_loss: 0.7362 - val_accuracy: 0.5077
Epoch 6/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.7301 - accuracy: 0.5093 - val_loss: 0.7234 - val_accuracy: 0.5000
Epoch 7/50
1830/1830 [==============================] - 99s 54ms/sample - loss: 0.7191 - accuracy: 0.5104 - val_loss: 0.7144 - val_accuracy: 0.5000
Epoch 8/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.7119 - accuracy: 0.5077 - val_loss: 0.7080 - val_accuracy: 0.5046
Epoch 9/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.7056 - accuracy: 0.5158 - val_loss: 0.7042 - val_accuracy: 0.5277
Epoch 10/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.7033 - accuracy: 0.5038 - val_loss: 0.7016 - val_accuracy: 0.5354
Epoch 11/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.7006 - accuracy: 0.5328 - val_loss: 0.6999 - val_accuracy: 0.5277
Epoch 12/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.6983 - accuracy: 0.5251 - val_loss: 0.6985 - val_accuracy: 0.5292
Epoch 13/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6966 - accuracy: 0.5306 - val_loss: 0.6981 - val_accuracy: 0.5031
Epoch 14/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.6960 - accuracy: 0.5202 - val_loss: 0.6977 - val_accuracy: 0.5277
Epoch 15/50
1830/1830 [==============================] - 115s 63ms/sample - loss: 0.6949 - accuracy: 0.5284 - val_loss: 0.6974 - val_accuracy: 0.5292
Epoch 16/50
1830/1830 [==============================] - 107s 58ms/sample - loss: 0.6940 - accuracy: 0.5219 - val_loss: 0.6972 - val_accuracy: 0.5323
Epoch 17/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6931 - accuracy: 0.5432 - val_loss: 0.6973 - val_accuracy: 0.5262
Epoch 18/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6925 - accuracy: 0.5546 - val_loss: 0.6970 - val_accuracy: 0.5246
Epoch 19/50
1830/1830 [==============================] - 101s 55ms/sample - loss: 0.6899 - accuracy: 0.5519 - val_loss: 0.6969 - val_accuracy: 0.5292
Epoch 20/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6877 - accuracy: 0.5672 - val_loss: 0.6973 - val_accuracy: 0.5354
Epoch 21/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6869 - accuracy: 0.5656 - val_loss: 0.6982 - val_accuracy: 0.5354
Epoch 22/50
1830/1830 [==============================] - 111s 61ms/sample - loss: 0.6869 - accuracy: 0.5628 - val_loss: 0.6983 - val_accuracy: 0.5262
Epoch 23/50
1830/1830 [==============================] - 109s 60ms/sample - loss: 0.6821 - accuracy: 0.5721 - val_loss: 0.6991 - val_accuracy: 0.5169
Epoch 24/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6794 - accuracy: 0.5902 - val_loss: 0.7009 - val_accuracy: 0.5338
Epoch 25/50
1830/1830 [==============================] - 101s 55ms/sample - loss: 0.6799 - accuracy: 0.5847 - val_loss: 0.6994 - val_accuracy: 0.5323
Epoch 26/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6765 - accuracy: 0.5951 - val_loss: 0.7010 - val_accuracy: 0.5277
Epoch 27/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.6784 - accuracy: 0.5836 - val_loss: 0.7045 - val_accuracy: 0.5323
Epoch 28/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6770 - accuracy: 0.5973 - val_loss: 0.7081 - val_accuracy: 0.5154
Epoch 29/50
1830/1830 [==============================] - 109s 60ms/sample - loss: 0.6693 - accuracy: 0.6049 - val_loss: 0.7057 - val_accuracy: 0.5323
Epoch 30/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6715 - accuracy: 0.6016 - val_loss: 0.7050 - val_accuracy: 0.5385
Epoch 31/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.6650 - accuracy: 0.6164 - val_loss: 0.7120 - val_accuracy: 0.5138
Epoch 32/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.6648 - accuracy: 0.6158 - val_loss: 0.7080 - val_accuracy: 0.5200
Epoch 33/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6614 - accuracy: 0.6213 - val_loss: 0.7248 - val_accuracy: 0.5015
Epoch 34/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6645 - accuracy: 0.6109 - val_loss: 0.7123 - val_accuracy: 0.5369
Epoch 35/50
1830/1830 [==============================] - 103s 57ms/sample - loss: 0.6588 - accuracy: 0.6235 - val_loss: 0.7146 - val_accuracy: 0.5185
Epoch 36/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6531 - accuracy: 0.6230 - val_loss: 0.7169 - val_accuracy: 0.5262
Epoch 37/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6506 - accuracy: 0.6393 - val_loss: 0.7217 - val_accuracy: 0.5108
Epoch 38/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6486 - accuracy: 0.6393 - val_loss: 0.7256 - val_accuracy: 0.5062
Epoch 39/50
1830/1830 [==============================] - 106s 58ms/sample - loss: 0.6438 - accuracy: 0.6557 - val_loss: 0.7290 - val_accuracy: 0.5062
Epoch 40/50
1830/1830 [==============================] - 105s 57ms/sample - loss: 0.6397 - accuracy: 0.6503 - val_loss: 0.7410 - val_accuracy: 0.5077
Epoch 41/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6405 - accuracy: 0.6514 - val_loss: 0.7290 - val_accuracy: 0.5046
Epoch 42/50
1830/1830 [==============================] - 101s 55ms/sample - loss: 0.6367 - accuracy: 0.6552 - val_loss: 0.7324 - val_accuracy: 0.5062
Epoch 43/50
1830/1830 [==============================] - 102s 56ms/sample - loss: 0.6309 - accuracy: 0.6645 - val_loss: 0.7450 - val_accuracy: 0.4954
Epoch 44/50
1830/1830 [==============================] - 121s 66ms/sample - loss: 0.6236 - accuracy: 0.6721 - val_loss: 0.7462 - val_accuracy: 0.5031
Epoch 45/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6191 - accuracy: 0.6820 - val_loss: 0.7432 - val_accuracy: 0.4954
Epoch 46/50
1830/1830 [==============================] - 108s 59ms/sample - loss: 0.6181 - accuracy: 0.6836 - val_loss: 0.7774 - val_accuracy: 0.5231
Epoch 47/50
1830/1830 [==============================] - 103s 56ms/sample - loss: 0.6104 - accuracy: 0.6929 - val_loss: 0.7717 - val_accuracy: 0.5138
Epoch 48/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.6038 - accuracy: 0.6902 - val_loss: 0.7660 - val_accuracy: 0.5169
Epoch 49/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.5979 - accuracy: 0.7022 - val_loss: 0.7569 - val_accuracy: 0.4954
Epoch 50/50
1830/1830 [==============================] - 104s 57ms/sample - loss: 0.5886 - accuracy: 0.7115 - val_loss: 0.7749 - val_accuracy: 0.5031
Preparing and testing system...
0016.HK
Nos. of train_data for ups: 144 and downs: 169
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 88, 178, 64)       640       
_________________________________________________________________
activation (Activation)      (None, 88, 178, 64)       0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 44, 89, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 44, 89, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 42, 87, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 43, 128)       0         
_________________________________________________________________
activation_1 (Activation)    (None, 21, 43, 128)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 21, 43, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 19, 41, 256)       295168    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 9, 20, 256)        0         
_________________________________________________________________
activation_2 (Activation)    (None, 9, 20, 256)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 9, 20, 256)        0         
_________________________________________________________________
flatten (Flatten)            (None, 46080)             0         
_________________________________________________________________
dense (Dense)                (None, 512)               23593472  
_________________________________________________________________
activation_3 (Activation)    (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
_________________________________________________________________
activation_4 (Activation)    (None, 1)                 0         
=================================================================
Total params: 23,963,649
Trainable params: 23,963,649
Non-trainable params: 0
_________________________________________________________________
accuracy: 53.99%
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


4. ####2020-01-09_0135--30-SEQ-5-PRED####

Included filled BB band

Preparing training data...
0823.HK
0101.HK
0012.HK
1997.HK
1109.HK
1113.HK
0016.HK
2007.HK


Preparing validating data...
0688.HK
0083.HK
0017.HK


Nos. of train_data for ups and downs: 824
Nos. of val_data for ups and downs: 450
Train on 1648 samples, validate on 900 samples
Epoch 1/50
1648/1648 [==============================] - 104s 63ms/sample - loss: 1.7138 - accuracy: 0.4964 - val_loss: 1.3826 - val_accuracy: 0.5000
Epoch 2/50
1648/1648 [==============================] - 100s 61ms/sample - loss: 1.2129 - accuracy: 0.4885 - val_loss: 1.0354 - val_accuracy: 0.5000
Epoch 3/50
1648/1648 [==============================] - 98s 60ms/sample - loss: 0.9395 - accuracy: 0.4806 - val_loss: 0.8594 - val_accuracy: 0.4867
Epoch 4/50
1648/1648 [==============================] - 102s 62ms/sample - loss: 0.8255 - accuracy: 0.4915 - val_loss: 0.7944 - val_accuracy: 0.5000
Epoch 5/50
1648/1648 [==============================] - 101s 61ms/sample - loss: 0.7767 - accuracy: 0.5146 - val_loss: 0.7601 - val_accuracy: 0.5000
Epoch 6/50
1648/1648 [==============================] - 103s 63ms/sample - loss: 0.7513 - accuracy: 0.4927 - val_loss: 0.7403 - val_accuracy: 0.5078
Epoch 7/50
1648/1648 [==============================] - 100s 61ms/sample - loss: 0.7341 - accuracy: 0.5006 - val_loss: 0.7278 - val_accuracy: 0.5000
Epoch 8/50
1648/1648 [==============================] - 113s 69ms/sample - loss: 0.7246 - accuracy: 0.5018 - val_loss: 0.7193 - val_accuracy: 0.5000
Epoch 9/50
1648/1648 [==============================] - 113s 68ms/sample - loss: 0.7160 - accuracy: 0.5152 - val_loss: 0.7122 - val_accuracy: 0.5000
Epoch 10/50
1648/1648 [==============================] - 105s 64ms/sample - loss: 0.7099 - accuracy: 0.5146 - val_loss: 0.7071 - val_accuracy: 0.5000
Epoch 11/50
1648/1648 [==============================] - 107s 65ms/sample - loss: 0.7065 - accuracy: 0.4830 - val_loss: 0.7035 - val_accuracy: 0.5011
Epoch 12/50
1648/1648 [==============================] - 108s 66ms/sample - loss: 0.7018 - accuracy: 0.5188 - val_loss: 0.7010 - val_accuracy: 0.5000
Epoch 13/50
1648/1648 [==============================] - 115s 70ms/sample - loss: 0.7012 - accuracy: 0.5061 - val_loss: 0.6999 - val_accuracy: 0.4878
Epoch 14/50
1648/1648 [==============================] - 106s 64ms/sample - loss: 0.6992 - accuracy: 0.5067 - val_loss: 0.6988 - val_accuracy: 0.5100
Epoch 15/50
1648/1648 [==============================] - 106s 64ms/sample - loss: 0.6985 - accuracy: 0.5127 - val_loss: 0.6981 - val_accuracy: 0.5000
Epoch 16/50
1648/1648 [==============================] - 102s 62ms/sample - loss: 0.6986 - accuracy: 0.4945 - val_loss: 0.6975 - val_accuracy: 0.5178
Epoch 17/50
1648/1648 [==============================] - 104s 63ms/sample - loss: 0.6960 - accuracy: 0.5249 - val_loss: 0.6972 - val_accuracy: 0.5133
Epoch 18/50
1648/1648 [==============================] - 103s 62ms/sample - loss: 0.6969 - accuracy: 0.5115 - val_loss: 0.6969 - val_accuracy: 0.5156
Epoch 19/50
1648/1648 [==============================] - 99s 60ms/sample - loss: 0.6955 - accuracy: 0.5370 - val_loss: 0.6968 - val_accuracy: 0.5211
Epoch 20/50
1648/1648 [==============================] - 100s 61ms/sample - loss: 0.6951 - accuracy: 0.5152 - val_loss: 0.6967 - val_accuracy: 0.5167
Epoch 21/50
1648/1648 [==============================] - 94s 57ms/sample - loss: 0.6912 - accuracy: 0.5467 - val_loss: 0.6963 - val_accuracy: 0.5133
Epoch 22/50
1648/1648 [==============================] - 86s 52ms/sample - loss: 0.6939 - accuracy: 0.5352 - val_loss: 0.6964 - val_accuracy: 0.5244
Epoch 23/50
1648/1648 [==============================] - 89s 54ms/sample - loss: 0.6892 - accuracy: 0.5504 - val_loss: 0.6961 - val_accuracy: 0.5267
Epoch 24/50
1648/1648 [==============================] - 92s 56ms/sample - loss: 0.6896 - accuracy: 0.5655 - val_loss: 0.6956 - val_accuracy: 0.5278
Epoch 25/50
1648/1648 [==============================] - 94s 57ms/sample - loss: 0.6896 - accuracy: 0.5613 - val_loss: 0.6959 - val_accuracy: 0.5222
Epoch 26/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6898 - accuracy: 0.5461 - val_loss: 0.6964 - val_accuracy: 0.5200
Epoch 27/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6865 - accuracy: 0.5619 - val_loss: 0.6966 - val_accuracy: 0.5122
Epoch 28/50
1648/1648 [==============================] - 87s 53ms/sample - loss: 0.6853 - accuracy: 0.5728 - val_loss: 0.6966 - val_accuracy: 0.5289
Epoch 29/50
1648/1648 [==============================] - 86s 52ms/sample - loss: 0.6835 - accuracy: 0.5728 - val_loss: 0.6973 - val_accuracy: 0.5222
Epoch 30/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6808 - accuracy: 0.5710 - val_loss: 0.6996 - val_accuracy: 0.5322
Epoch 31/50
1648/1648 [==============================] - 90s 55ms/sample - loss: 0.6814 - accuracy: 0.5680 - val_loss: 0.6980 - val_accuracy: 0.5289
Epoch 32/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6760 - accuracy: 0.5910 - val_loss: 0.6992 - val_accuracy: 0.5333
Epoch 33/50
1648/1648 [==============================] - 93s 57ms/sample - loss: 0.6763 - accuracy: 0.5862 - val_loss: 0.6993 - val_accuracy: 0.5367
Epoch 34/50
1648/1648 [==============================] - 88s 54ms/sample - loss: 0.6718 - accuracy: 0.5934 - val_loss: 0.7016 - val_accuracy: 0.5444
Epoch 35/50
1648/1648 [==============================] - 88s 53ms/sample - loss: 0.6700 - accuracy: 0.6056 - val_loss: 0.7013 - val_accuracy: 0.5078
Epoch 36/50
1648/1648 [==============================] - 87s 53ms/sample - loss: 0.6659 - accuracy: 0.6080 - val_loss: 0.7034 - val_accuracy: 0.5356
Epoch 37/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6631 - accuracy: 0.6080 - val_loss: 0.7049 - val_accuracy: 0.5356
Epoch 38/50
1648/1648 [==============================] - 85s 51ms/sample - loss: 0.6574 - accuracy: 0.6129 - val_loss: 0.7060 - val_accuracy: 0.5278
Epoch 39/50
1648/1648 [==============================] - 88s 54ms/sample - loss: 0.6554 - accuracy: 0.6214 - val_loss: 0.7089 - val_accuracy: 0.5211
Epoch 40/50
1648/1648 [==============================] - 88s 53ms/sample - loss: 0.6598 - accuracy: 0.6117 - val_loss: 0.7093 - val_accuracy: 0.5333
Epoch 41/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6541 - accuracy: 0.6141 - val_loss: 0.7107 - val_accuracy: 0.5189
Epoch 42/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6535 - accuracy: 0.6341 - val_loss: 0.7098 - val_accuracy: 0.5144
Epoch 43/50
1648/1648 [==============================] - 91s 55ms/sample - loss: 0.6498 - accuracy: 0.6208 - val_loss: 0.7134 - val_accuracy: 0.5267
Epoch 44/50
1648/1648 [==============================] - 90s 54ms/sample - loss: 0.6401 - accuracy: 0.6408 - val_loss: 0.7135 - val_accuracy: 0.5067
Epoch 45/50
1648/1648 [==============================] - 88s 54ms/sample - loss: 0.6405 - accuracy: 0.6383 - val_loss: 0.7174 - val_accuracy: 0.5289
Epoch 46/50
1648/1648 [==============================] - 88s 53ms/sample - loss: 0.6400 - accuracy: 0.6414 - val_loss: 0.7190 - val_accuracy: 0.5089
Epoch 47/50
1648/1648 [==============================] - 83s 50ms/sample - loss: 0.6384 - accuracy: 0.6432 - val_loss: 0.7277 - val_accuracy: 0.5267
Epoch 48/50
1648/1648 [==============================] - 84s 51ms/sample - loss: 0.6313 - accuracy: 0.6553 - val_loss: 0.7209 - val_accuracy: 0.5256
Epoch 49/50
1648/1648 [==============================] - 83s 51ms/sample - loss: 0.6236 - accuracy: 0.6608 - val_loss: 0.7233 - val_accuracy: 0.5178
Epoch 50/50
1648/1648 [==============================] - 87s 53ms/sample - loss: 0.6136 - accuracy: 0.6681 - val_loss: 0.7302 - val_accuracy: 0.5222
Preparing and testing system...
0017.HK
Nos. of train_data for ups: 148 and downs: 178
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 88, 178, 64)       640       
_________________________________________________________________
activation (Activation)      (None, 88, 178, 64)       0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 44, 89, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 44, 89, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 42, 87, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 43, 128)       0         
_________________________________________________________________
activation_1 (Activation)    (None, 21, 43, 128)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 21, 43, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 19, 41, 256)       295168    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 9, 20, 256)        0         
_________________________________________________________________
activation_2 (Activation)    (None, 9, 20, 256)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 9, 20, 256)        0         
_________________________________________________________________
flatten (Flatten)            (None, 46080)             0         
_________________________________________________________________
dense (Dense)                (None, 512)               23593472  
_________________________________________________________________
activation_3 (Activation)    (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
_________________________________________________________________
activation_4 (Activation)    (None, 1)                 0         
=================================================================
Total params: 23,963,649
Trainable params: 23,963,649
Non-trainable params: 0
_________________________________________________________________
accuracy: 54.60%

