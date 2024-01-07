#!/bin/bash
read -p "파일 이름을 입력하세요: " FILE_NAME
UPLOAD_FOLDER="19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6"
gdrive upload --parent $UPLOAD_FOLDER $FILE_NAME --share