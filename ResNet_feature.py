from __future__ import print_function
import sys
import os as os
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[2]
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout,Dense, Reshape,Flatten, LeakyReLU, add, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras import backend as K
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob as glob
import numpy as np
import random

#from scipy.stats import threshold

######### Set path where data (dense flow and raw frames) are stored
path=sys.argv[3] #"./data_pre/"
#path='/Neutron9/bharat.b/data_pre/'


######### Set path where models must be stored
path2sav=sys.argv[4] #"./folder5/"
#path2sav='/Neutron9/bharat.b/folder5/'

name=sys.argv[1] #"Huji_optical_flow_30_frame"
suffix='ver1'
fr_rate=30
if not(os.path.isdir(path2sav)):
	os.mkdir(path2sav)

if not(os.path.isdir(path2sav+name)):
	os.mkdir(path2sav+name)

if not(os.path.isdir(path2sav+name+'/models_'+suffix)):
	os.mkdir(path2sav+name+'/models_'+suffix)

'''
if not(os.path.isdir(path2sav+name+'/mean_std_'+suffix)):
	os.mkdir(path2sav+name+'/mean_std_'+suffix)
'''

########## Set parameters here
inp_sz=(36,64)
drop=0
n_class=3 #TODO: for Huji right now
epoch=200
opti='adam'
los=keras.losses.categorical_crossentropy
no_layer=[16,32,64,128]
no_auto=20
no_sp_per_vid=20
batch_sz=2000

####Set type here, x for flow_x, y for flow_y and <blank> for frame
if(len(sys.argv)==6):
	typ=sys.argv[5] #or 'y'
else:
	typ=''

#Architecture

def add_common_layers(y):
	y = BatchNormalization(axis = 1)(y)
	y = LeakyReLU()(y)
	return y

def residual_block(y, nb_channels, _strides=(1,1), _project_shortcut=False):
	shortcut = y

	# down_sampling is performed with a stride of 2
	y = Conv2D(nb_channels, kernel_size=(3,3), strides=_strides, padding='same', data_format="channels_first")(y)
	y = add_common_layers(y)

	y = Conv2D(nb_channels, kernel_size=(3,3), strides=(1,1), padding='same', data_format="channels_first")(y)
	y = BatchNormalization(axis = 1)(y)
	# identity shortcuts used directly when the input and output are of the same dimensions
	if (_project_shortcut or _strides != (1, 1)):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
		shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', data_format="channels_first")(shortcut)
		shortcut = BatchNormalization(axis = 1)(shortcut)

	y = add([shortcut, y])
	y = LeakyReLU()(y)

	return(y)

# Define layer 1
input_img1 = Input(shape=(1, inp_sz[0], inp_sz[1]))
x1 = Conv2D(no_layer[0], (3, 3), padding="same", data_format="channels_first")(input_img1)
x1 = add_common_layers(x1)
x1 = MaxPooling2D((2,2), padding='valid', data_format = 'channels_first')(x1)
for i in range(3):
	project_shortcut = True if i == 0 else False
	x1 = residual_block(x1, no_layer[1], _project_shortcut=project_shortcut)

for i in range(4):
	# down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
	strides = (2, 2) if i == 0 else (1, 1)
	x1 = residual_block(x1, no_layer[2], _strides=strides)

# for i in range(6):
# 	strides = (2, 2) if i == 0 else (1, 1)
# 	x1 = residual_block(x1, no_layer[3], _project_shortcut=project_shortcut)
#print(x1.shape)
# x1 = GlobalAveragePooling2D(data_format="channels_first")(x1)
# print(x1.shape)
# x1 = Dense()(x1)
# print(x1.shape)
# exit()
x1 = Flatten(data_format='channels_first')(x1)
x1 = Dense(500, activation='relu')(x1)
x1 = Dense(100, activation='relu')(x1)
# Loss function for classification 
result_class = Dense(n_class, activation='softmax')(x1)


model = Model(inputs=input_img1, outputs=result_class)
#print(model.summary())
model.compile(optimizer=opti, loss=los, metrics=['accuracy'])
json_string=model.to_json()
open(path2sav+'model_temp.json','w').write(json_string)


#load data
# total_file contains framewise optical flow field: (frames, dim x, dim y)
# file_class contains one_hot encoding of framewise class labels: (frames, num of classes)

folder=sorted(glob.glob(path+name+'/*'))
total_file=[]
file_class=[]
ind = []
path_gt = './Huji/'

print('start loading data')
print(path+name+'/feat_'+typ+'.npy')
if os.path.exists(path+name+'/feat_'+typ+'.npy'):
	print('skipping extracting feature '+typ)
	total_file=np.load(path+name+'/feat_'+typ+'.npy')
	file_class=np.load(path+name+'/class_labels.npy')
	ind = np.load(path+name+'/ind.npy')
else:
	exit()
	for i in folder:
		print(i)
		print(typ)
		name_folder = i[49:-3]   
		temp=sorted(glob.glob(i+'/'+'*'+typ+'*.npy'))
		file=[]
		for opt_flow_frames in temp:
			opt_flow = np.load(opt_flow_frames)
			if (file==[]):
				file = opt_flow
			else:
				file = np.concatenate((file,opt_flow))
		n_frames = file.shape[0]
		path_gt_v = path_gt+name_folder+'.mat'
		gt = scipy.io.loadmat(path_gt_v)['gt'][:n_frames]
		# normalize gt, with class labels 0, 1, 2, 3...
		# in Huji ground truth, we only have class labels 1,2,3
		gt = gt - 1
		ind.append(gt.shape[0])
		gt = keras.utils.to_categorical(np.squeeze(gt),n_class)
	    
	    # Put everything in one file
		if (total_file==[]):
			total_file = file
			file_class = gt
		else:
			total_file = np.concatenate((total_file,file))
			file_class = np.concatenate((file_class,gt))
	np.save(path+name+'/feat_'+typ, total_file)
	np.save(path+name+'/class_labels', file_class)
	np.save(path+name+'/ind', np.array(ind))

# train/test split 
split = np.sum(ind[int(np.round(len(ind)*0.7))])
#TODO: check every class has samples in training set

train=[:split,:,:]
train_labels = [:split, :]
test=[split:,:,:]
test_labels=[split,:]


# Training
model = model_from_json(open(path2sav+'model_temp.json').read())
model.compile(optimizer=opti,loss=los,metrics=['accuracy'])

model.fit(train, train_labels, batch_size=batch_sz, verbose=1, epochs=epoch, validation_split=0.1)
score = model.evaluate(test, test_labels, verbose=0)
model.save_weights(path2sav+name+'/models_'+suffix+'/model_'+typ+'.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])









