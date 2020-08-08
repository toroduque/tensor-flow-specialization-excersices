# run `make getfiles` to download the training and test images.

getfiles: getfiles.training_data getfiles.test_data unzip_files

getfiles.training_data:
	wget --no-check-certificate \
	https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O ./tmp/horse-or-human.zip

getfiles.test_data:
	wget --no-check-certificate \
	https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O ./tmp/validation-horse-or-human.zip

unzip_files:
	python3 unzip_files.py ./