#!/bin/bash
gdrive list --query "trashed = false and '19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6' in parents"
read -p "파일 아이디를 입력하세요: " FILE_ID
gdown "https://drive.google.com/uc?id=${FILE_ID}"