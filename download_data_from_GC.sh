#!/bin/bash
gdrive list --query "trashed = false and '19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6' in parents"
read -p "파일 아이디를 입력하세요: " FILE_ID
# 해당 파일을 모든 사람에게 읽기 전용으로 공개한다.
# gdrive share $FILE_ID

# 다운로드후 파일 이름을 바꿀 수 있다.
# output_file="filename_after_download"
# download_dir="/data/ephemeral"
# gdown "https://drive.google.com/uc?id=${file_id}" -O "${download_dir}/${output_file}"
gdown "https://drive.google.com/uc?id=${FILE_ID}"