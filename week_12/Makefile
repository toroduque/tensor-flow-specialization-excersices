# run `make getfiles` to download the 'Is sarcasm?' data set

getfiles: create_tmp_dir get_training_data get_assignment_data

create_tmp_dir:
	mkdir ./tmp

get_training_data:
	wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
    -O ./tmp/irish-lyrics-eof.txt
