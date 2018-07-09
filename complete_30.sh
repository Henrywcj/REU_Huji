gpu=0

# Extract optical flow
mkdir data_pre
search_dir="./Huji"
for entry in "$search_dir"/*.MP4
do
  echo "$entry"
  python3 opti.py $entry 30 "./data_pre/"
done

for entry in "$search_dir"/*.mp4
do
  echo "$entry"
  python3 opti.py $entry 30 "./data_pre/"
done


# Train spatial autoencoders for flow_x, flow_y and frame respectivly
python3 train_2D_mult_auto_ver4.py "Huji_optical_flow_30_frame" $gpu "./data_pre/" "./folder5/" "x"
python3 train_2D_mult_auto_ver4.py "Huji_optical_flow_30_frame" $gpu "./data_pre/" "./folder5/" "y"

# Extract features from flow_x stream
ser_dir="./data_pre/Huji_optical_flow_30_frame"
name="Huji_optical_flow_30_frame"
type="x"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python3 test_2D_conv.py $name $f $gpu "./data_pre/" "./folder5/" $type
done

# Extract features from flow_y stream
name="Huji_optical_flow_30_frame"
type="y"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python3 test_2D_conv.py $name $f $gpu "./data_pre/" "./folder5/" $type
done

# Train LSTM autoencoder for flow_x, flow_y and frame respectivly
python3 lstm_2D_conv_ver4.py "Huji_optical_flow_30_frame" $gpu './data_pre/' './folder5/' 'x'
python3 lstm_2D_conv_ver4.py "Huji_optical_flow_30_frame" $gpu './data_pre/' './folder5/' 'y'

# Extract LSTM features for flow_x
ser_dir="./data_pre/Huji_optical_flow_30_frame"
name="Huji_optical_flow_30_frame"
type="x"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python3 test_lstm.py $name $f $gpu './folder5/' $type
done

# Extract LSTM features for flow_y
ser_dir="./data_pre/Huji_optical_flow_30_frame"
name="Huji_optical_flow_30_frame"
type="y"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python3 test_lstm.py $name $f $gpu './folder5/' $type
done


