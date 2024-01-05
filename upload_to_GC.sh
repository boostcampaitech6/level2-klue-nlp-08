# 공식패키지가 없기때문에 깃헙에서 gdrive 다운로드 받아준다.
# wget https://github.com/BugCode1/gdrive/releases/download/2.1.2/gdrive_2.1.2_linux_386.tar.gz
# tar -xvf gdrive_2.1.2_linux_386.tar.gz
# mv gdrive /usr/local/bin/gdrive

# 공식패키지가 아니어서 그런지 10월부터 접근권한 문제가 생겨서 2.1.2 버전을 내서 해결하시긴 했지만 임시방편이라서 좀 번거로움.
# https://github.com/BugCode1/gdrive 를 사용했음. 
# gdrive list
# 링크가 출력될 것이다. 들어가서 본인 구글 드라이브에 로그인하면 아래 url로 넘어갈것이다.
# http://localhost:1/?state=state&code=여기_부분_복사&scope=https://www.googleapis.com/auth/drive
# 복사한 부분을 Enter verification code에 넣어준다. 그리고 따로 어딘가에 저장해두는게 좋을 듯.

file_name="업로드할 파일 이름"
# 공유 폴더 링크 https://drive.google.com/drive/folders/19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6?usp=drive_link
upload_folder="19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6"
gdrive upload --parent upload_folder file_name --share