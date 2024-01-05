********************************************************

download_from_GC.sh 실행 전에 해야 할것.
먼저 구글 드라이브에서 다운로드받을 파일의 공유 옵션을 "링크가 있는 모든 사용자"로 하고 공유에서 링크 복사를 누른다.
https://drive.google.com/file/d/이_부분이_파일_ID/view?usp=sharing
아래 명령어를 순차적으로 실행한다.
pip install gdown
chmod +x download_from_GC.sh
이후 아래 명령어를 치면 실행된다. 인자로 파일 ID를 넘겨줘야 한다.
./download_from_GC.sh 파일ID

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
이후 아래 명령어를 치면 실행된다. 인자로 파일명을 넘겨줘야 한다.
./upload_to_GC.sh 파일명

********************************************************