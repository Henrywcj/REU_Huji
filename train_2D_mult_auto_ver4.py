## Input format: code NameofTrainingSeq GpuNo. Path2Data Path2sav Type

from __future__ import print_function
import sys
import os as os
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[2]
from keras.layers import Input, Conv2D, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout,Dense, Reshape,Flatten, Deconvolution2D
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

name=sys.argv[1] #"Huji"
suffix='ver4_2'
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
epoch=200
opti='adam'
no_layer=[8,18]
no_auto=20
no_sp_per_vid=20
batch_sz=2000

####Set type here, x for flow_x, y for flow_y and <blank> for frame
if(len(sys.argv)==6):
	typ=sys.argv[5] #or 'y'
else:
	typ=''


#Architecture

# Define layer 1
input_img1 = Input(shape=(1, inp_sz[0], inp_sz[1]))
x1 = Conv2D(no_layer[0], (3, 3), padding="same", data_format="channels_first", activation="tanh")(input_img1)
x1 = Conv2D(no_layer[1], (5, 5), padding="same", data_format="channels_first", activation="tanh")(x1)
x1 = BatchNormalization(axis=1) (x1)
y1 = MaxPooling2D((2,2), padding='valid', data_format = 'channels_first')(x1)
x1 = Dropout(drop)(y1)
x1 = Conv2D(no_layer[1], (5, 5), padding="same", data_format="channels_first", activation="tanh")(x1)
x1 = Conv2D(no_layer[0], (3, 3), padding="same", data_format="channels_first", activation="tanh")(x1)
x1 = BatchNormalization(axis=1) (x1)
x1 = UpSampling2D((2, 2), data_format = 'channels_first')(x1)
decoded1 = Conv2D(1, (3, 3), padding="same", activation="sigmoid", data_format="channels_first")(x1)

#Make layer1
autoencoder1 = Model(inputs=input_img1, outputs=decoded1)
autoencoder1.compile(optimizer=opti, loss='binary_crossentropy')
encoder1 = Model(inputs=input_img1, outputs=y1)
json_string = autoencoder1.to_json()
open(path2sav+'autoencoder1_temp.json', 'w').write(json_string)


# Define layer2
input_img2 = Input(shape=(encoder1.output_shape[1], encoder1.output_shape[2], encoder1.output_shape[3]))
x2 = Conv2D(no_layer[0], (3, 3), padding="same", data_format="channels_first", activation="tanh")(input_img2)
x2 = Conv2D(no_layer[1], (5, 5), padding="same", data_format="channels_first", activation="tanh")(x2)
x2 = BatchNormalization(axis=1) (x2)
y2 = MaxPooling2D((2,2), padding='valid', data_format = 'channels_first')(x2)
x2 = Dropout(drop)(y2)
x2 = Conv2D(no_layer[1], (5, 5), padding="same", data_format="channels_first", activation="tanh")(x2)
x2 = Conv2D(no_layer[0], (3, 3), padding="same", data_format="channels_first", activation="tanh")(x2)
x2 = BatchNormalization(axis=1)(x2)
x2 = UpSampling2D((2, 2), data_format = 'channels_first')(x2)
decode2 = Conv2D(no_layer[1], (3, 3), padding="same", activation="sigmoid", data_format="channels_first")(x2)

#Make layer2
autoencoder2 = Model(inputs=input_img2, outputs=decode2)
autoencoder2.compile(optimizer=opti, loss='mean_squared_error')
encoder2 = Model(inputs=input_img2, outputs=y2)
json_string = autoencoder2.to_json()
open(path2sav+'autoencoder2_temp.json', 'w').write(json_string)

# Define layer3
input_img3 = Input(shape=(encoder2.output_shape[1], encoder2.output_shape[2], encoder2.output_shape[3]))
x3 = Flatten(data_format = 'channels_first')(input_img3)
x3 = Dense(500, activation='tanh')(x3)
y3 = Dense(100, activation='tanh')(x3)
x3 = Dense(500, activation='tanh')(y3)
x3 = Dense(encoder2.output_shape[1]*encoder2.output_shape[2]*encoder2.output_shape[3], activation='tanh')(x3)
decode3 = Reshape((encoder2.output_shape[1], encoder2.output_shape[2], encoder2.output_shape[3])) (x3)

#Make layer3
autoencoder3 = Model(inputs=input_img3, outputs=decode3)
autoencoder3.compile(optimizer=opti, loss='mean_squared_error')
encoder3 = Model(inputs=input_img3, outputs=y3)
json_string = autoencoder3.to_json()
open(path2sav+'autoencoder3_temp.json', 'w').write(json_string)

folder=sorted(glob.glob(path+name+'/*'))

# Load file names and randomly shuffle
file=[]
for i in folder:
	print(i)
	if(file==[]):
		temp=glob.glob(i+'/'+'*'+typ+'*.npy')
		random.shuffle(temp)
		file=temp[0:no_sp_per_vid]
	else:
		temp=glob.glob(i+'/*'+typ+'*.npy')
		random.shuffle(temp)
		file.extend(temp[0:no_sp_per_vid])

random.shuffle(file)
#l=len(file)-1
l=len(file)
mean=[]
std=[]
print('starting '+name+' '+typ)


for k in range(no_auto):
	if os.path.exists(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+typ+'_'+str(k).zfill(2)+'.h5'):
		print('skipping '+typ+' '+str(k))
		continue
	
	# Load each decoder and encoder layer as autoencoder1,2,3
	autoencoder1 = model_from_json(open(path2sav+'autoencoder1_temp.json').read())
	autoencoder1.compile(optimizer=opti, loss='binary_crossentropy')
	
	autoencoder2 = model_from_json(open(path2sav+'autoencoder2_temp.json').read())
	autoencoder2.compile(optimizer=opti, loss='mean_squared_error')
	
	autoencoder3 = model_from_json(open(path2sav+'autoencoder3_temp.json').read())
	autoencoder3.compile(optimizer=opti, loss='mean_squared_error')
	fin=[]
	
	# Load data from each file
	for i in range(k*l//no_auto,(k+1)*l//no_auto):
		data=np.load(file[i])
		for ip,io in enumerate(data):
			data[ip]=io-io.mean()
			data[ip]=data[ip]//data[ip].std()
		
		data=np.reshape(data, (fr_rate, 1, inp_sz[0],inp_sz[1]))
		if(fin==[]):
			fin=data
		else:
			fin=np.append(fin,data,0)
	'''
	if(mean==[]):
		mean=np.mean(np.mean(fin,1),1)
		std=np.std(np.std(abs(fin),1),1)
	else:
		mean=np.append(mean,np.mean(np.mean(fin,1),1),0)
		std=np.append(std,np.std(np.std(abs(fin),1),1),0)
	
	fin=np.subtract(fin,mean[k])
        fin=np.divide(fin,std[k])
	'''
	print(len(fin))
	# Train each successive layer
	history1=autoencoder1.fit(fin,fin, nb_epoch=epoch*2,batch_size=batch_sz,verbose=0)
	q1 = K.function([autoencoder1.layers[0].input,K.learning_phase()],[autoencoder1.layers[4].output])
	out1=q1([fin,0])[0]
	history2=autoencoder2.fit(out1,out1,nb_epoch=epoch,batch_size=batch_sz,verbose=0)
	q1 = K.function([autoencoder2.layers[0].input,K.learning_phase()],[autoencoder2.layers[4].output])
	out2=q1([out1,0])[0]
	history3=autoencoder3.fit(out2,out2,nb_epoch=epoch*2,batch_size=batch_sz,verbose=0)
	
	#Fine tune
	input_img = Input(shape=(1, inp_sz[0], inp_sz[1]))
	x=Conv2D(no_layer[0], (3, 3), padding="same", data_format='channels_first', weights=autoencoder1.layers[1].get_weights(), activation="tanh")(input_img)
	x=Conv2D(no_layer[1], (5, 5), padding="same", data_format='channels_first', weights=autoencoder1.layers[2].get_weights(), activation="tanh")(x)
	x=BatchNormalization(axis=1, weights=autoencoder1.layers[3].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=MaxPooling2D((2,2), padding='valid', data_format = 'channels_first')(x)
	
	x=Conv2D(no_layer[0], (3, 3), padding="same", data_format='channels_first', weights=autoencoder2.layers[1].get_weights(), activation="tanh")(x)
	x=Conv2D(no_layer[1], (5, 5), padding="same", data_format='channels_first', weights=autoencoder2.layers[2].get_weights(), activation="tanh")(x)
	x=BatchNormalization(axis=1, weights=autoencoder2.layers[3].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=MaxPooling2D((2,2), padding='valid', data_format = 'channels_first')(x)
		
	x = Flatten(data_format = 'channels_first')(x)
	x = Dense(500, activation='tanh', weights=autoencoder3.layers[2].get_weights())(x)
	y = Dense(100, activation='tanh', weights=autoencoder3.layers[3].get_weights())(x)
	x = Dense(500, activation='tanh', weights=autoencoder3.layers[4].get_weights())(y)
	x = Dense(encoder2.output_shape[1]*encoder2.output_shape[2]*encoder2.output_shape[3], weights=autoencoder3.layers[5].get_weights(), activation='tanh')(x)
	x = Reshape((encoder2.output_shape[1],encoder2.output_shape[2],encoder2.output_shape[3])) (x)
	

	x=Conv2D(no_layer[1], (5, 5), padding="same", data_format='channels_first', weights=autoencoder2.layers[6].get_weights(), activation='tanh')(x)
	x=Conv2D(no_layer[0], (3, 3), padding="same", data_format='channels_first', weights=autoencoder2.layers[7].get_weights(), activation='tanh')(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=BatchNormalization(axis=1, weights=autoencoder2.layers[8].get_weights())(x)
	x=UpSampling2D((2, 2), data_format = 'channels_first')(x)
	x=Conv2D(no_layer[1], (3, 3), padding='same', data_format='channels_first', weights=autoencoder2.layers[10].get_weights(), activation='tanh')(x)
	x=Conv2D(no_layer[1], (5, 5), padding='same', data_format='channels_first', weights=autoencoder1.layers[6].get_weights(), activation='tanh')(x)
	x=Conv2D(no_layer[0], (3, 3), padding='same', data_format='channels_first', weights=autoencoder1.layers[7].get_weights(), activation='tanh')(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=BatchNormalization(axis=1, weights=autoencoder1.layers[8].get_weights())(x)
	x=UpSampling2D((2, 2), data_format = 'channels_first')(x)
	decode=Conv2D(1, (3, 3), padding='same', data_format='channels_first', weights=autoencoder1.layers[10].get_weights(), activation='sigmoid')(x)
	
	model = Model(input_img, decode)
	model.compile(optimizer=opti, loss='binary_crossentropy')
	history = model.fit(fin, fin,nb_epoch=epoch,batch_size=batch_sz,verbose=0)
	model.save_weights(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+typ+'_'+str(k).zfill(2)+'.h5')
	print(k)

json_string = model.to_json()
open(path2sav+'pretrained_conv_auto_'+suffix+'.json', 'w').write(json_string)

#np.save(path2sav+name+'/mean_std_'+suffix+'/mean_'+type+'.npy',mean)
#np.save(path2sav+name+'/mean_std_'+suffix+'/std_'+type+'.npy',std)
print(typ)
