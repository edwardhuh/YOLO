#!/bin/bash

set -e

if which python3 > /dev/null; then
	PYTHON=python3
else
	PYTHON=python
fi
echo "WARNING: PLEASE RUN THIS IN THE data directory!!"

echo "** Install requirements"
# "gdown" package to interface with GoogleDrive
pip3 install --user gdown > /dev/null

# make sure to download dataset files to "data"
mkdir -p $(dirname $0)/raw
pushd $(dirname $0)/raw > /dev/null

get_file()
{
	# ensure the files dont exist
	if [[ -f $2 ]]; then
		echo Skipping $2
	else
		echo Downloading $2...
		python3 -m gdown.cli $1
	fi
}

echo "** Download dataset files"
get_file https://drive.google.com/uc?id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y CrowdHuman_train01.zip
get_file https://drive.google.com/uc?id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO CrowdHuman_val.zip
# test data is not needed...
# get_file https://drive.google.com/uc?id=1tQG3E_RrRI4wIGskorLTmDiWHH2okVvk CrowdHuman_test.zip
get_file https://drive.google.com/u/0/uc?id=1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3 annotation_train.odgt
get_file https://drive.google.com/u/0/uc?id=10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL annotation_val.odgt


# unzip image files (ignore CrowdHuman_test.zip for now)
echo "** Unzip dataset files"
for f in CrowdHuman_train01.zip CrowdHuman_val.zip ; do
  unzip -n ${f}
done

echo "** Create data file structure"
mkdir ../processed/resized
mkdir ../processed/resized_one_file

