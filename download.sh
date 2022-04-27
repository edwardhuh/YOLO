/bin/bash
url = https://drive.google.com/uc?export=download&confirm=no_antivirus&id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y

#this downloads the zip file that contains the data
curl ${url} -o CrowdHuman_train01.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip CrowdHuman_train01.zip
# this cleans up the zip file, as we will no longer use it
rm CrowdHuman_train01.zip

# echo downloaded data
