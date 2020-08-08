# run `make getfiles` to download the training and test images.

getfiles: get_training_data unzip_files

get_training_data:
	wget --no-check-certificate \
  	https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
  	-O ./tmp/cats_and_dogs_filtered.zip

unzip_files:
	python3 unzip_files.py ./