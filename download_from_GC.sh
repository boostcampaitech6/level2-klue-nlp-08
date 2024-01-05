#!/bin/bash
FILE_ID="$1"
# output_file="filename_after_download"
# download_dir="/data/ephemeral"
# gdown "https://drive.google.com/uc?id=${file_id}" -O "${download_dir}/${output_file}"
gdown "https://drive.google.com/uc?id=${FILE_ID}"