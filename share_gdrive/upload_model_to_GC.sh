#!/bin/bash
read -p "파일 이름을 입력하세요: " FILE_NAME
UPLOAD_FOLDER="1muMepSTJom6qv-9TxdFTBEzOnJxRrQhW"
gdrive upload --parent $UPLOAD_FOLDER $FILE_NAME --share