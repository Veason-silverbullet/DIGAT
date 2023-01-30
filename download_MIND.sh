# Download and extract MIND-large dataset
cd ..
mkdir MIND-large
cd MIND-large
wget -O train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
unzip train.zip -d train
wget -O dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
unzip dev.zip -d dev
wget -O test.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip test.zip -d test


# Download and extract MIND-small dataset
cd ..
mkdir MIND-small
cd MIND-small
wget -O train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
unzip train.zip -d train
wget -O dev.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip dev.zip -d dev


# Preprocess datasets for experiments
cd ../DIGAT
python prepare_MIND_dataset.py
