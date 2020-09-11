# run `make getfiles` to download the pre-trained model

getfiles: create_tmp_dir get_pretrained_weights get_training_data unzip_files

get_pretrained_weights:
	wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O ./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

create_tmp_dir:
	mkdir ./tmp

get_training_data:
	wget --no-check-certificate \
  	https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
  	-O ./tmp/cats_and_dogs_filtered.zip

unzip_files:
	python3 unzip_files.py ./
