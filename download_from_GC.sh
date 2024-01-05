# 실행방법
# 먼저 구글 드라이브에서 해당 파일의 공유 옵션을 "링크가 있는 모든 사용자"로 하고
# pip install gdown
# chmod +x download_from_GC.sh
# ./download_from_GC.sh

file_id="파일_ID는_URL_마지막부분"
# output_file="filename_after_download"
# download_dir="/data/ephemeral"
# gdown "https://drive.google.com/uc?id=${file_id}" -O "${download_dir}/${output_file}"
gdown "https://drive.google.com/uc?id=${file_id}"

