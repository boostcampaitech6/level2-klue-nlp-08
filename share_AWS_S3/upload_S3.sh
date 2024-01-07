#!/bin/bash
read -p "Is it model or data?[m/d]: " FOLDER
read -p "Input file name you upload: " FILE_NAME
if [ $FOLDER = "m" ]; then
  aws s3 cp $FILE_NAME "s3://boostcampbucket/model/${FILE_NAME}"
elif [ $FOLDER = "d" ]; then
  aws s3 cp $FILE_NAME "s3://boostcampbucket/data/${FILE_NAME}"
else
  echo "first input must be only m or d"