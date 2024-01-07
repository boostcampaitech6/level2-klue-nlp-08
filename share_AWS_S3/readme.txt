********************************************************

AWSCLI를 다운받아서 사용해야 한다. 아래 명령어를 순차적으로 실행한다.
wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
unzip awscli-exe-linux-x86_64.zip
./aws/install

설치가 잘 되었는지 확인한다.
aws --version
ID와 비밀번호를 설정해야 한다.
aws configure
ID와 비밀번호 이외에는 그냥 엔터를 누르면 된다.

쉘 스크립트에 실행권한을 부여해줘야 한다.
chmod +x download_S3.sh 혹은 chmod +x upload_S3.sh
이후 아래 명령어를 치면 실행된다.
./download_S3.sh 혹은 ./upload_S3.sh
파일이 모델인지 데이터셋인지를 입력하고, 파일 이름을 입력하면 된다.

********************************************************