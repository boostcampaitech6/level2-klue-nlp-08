********************************************************

구글 드라이브 공용 데이터 저장 폴더.
https://drive.google.com/drive/folders/19fTYxwpJ6rIuSKcVWg1F77PYascRhA_6?usp=sharing

********************************************************

upload_to_GC.sh 실행 전에 해야 할것.
공식패키지가 없기때문에 깃헙에서 구글 드라이브 연동 프로그램을 받아야 한다. 아래 명령어를 순차적으로 실행한다.
wget https://github.com/BugCode1/gdrive/releases/download/2.1.2/gdrive_2.1.2_linux_386.tar.gz
tar -xvf gdrive_2.1.2_linux_386.tar.gz
mv gdrive /usr/local/bin/gdrive
gdrive list
링크가 출력될 것이다. 들어가서 본인 구글 드라이브에 로그인하면 아래 url로 넘어갈것이다.
http://localhost:1/?state=state&code=여기_부분_복사&scope=https://www.googleapis.com/auth/drive
복사한 부분을 Enter verification code에 넣어준다. 그리고 따로 어딘가에 저장해두는게 좋을 듯.
아래 명령어로 실행가능하게 만들어준다.
chmod +x upload_to_GC.sh
이후 아래 명령어를 치면 실행된다.
./upload_data_to_GC.sh
파일명을 입력하라고 나올 것이다. 쉘 스크립트와 같은 위치에 파일이 있지 않다면 디렉토리를 넣어주자.

********************************************************

download_from_GC.sh 실행 전에 해야 할것.
위의 gdrive 다운로드 밑 인증과정을 거쳐야 한다.
구글 드라이브의 해당 파일이 공개되어 있지 않다면 바꿔줘야 하지만 업로드 스크립트로 업로드 했다면 기본적으로 공개되어 있을 것이다.
아래 명령어를 순차적으로 실행한다.
pip install gdown
chmod +x download_from_GC.sh
이후 아래 명령어를 치면 실행된다.
./download_data_from_GC.sh 혹은 ./download_model_from_GC.sh
파일들의 리스트가 나올 것이다. 다운받을 파일의 ID를 입력해주면 된다.

********************************************************