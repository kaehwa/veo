# Python 3.10 slim 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /generate_video

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# outputs 폴더 생성
RUN mkdir -p /generate_video/outputs

# outputs 폴더를 volume로 설정 (로컬 마운트 가능)
VOLUME ["/generate_video/outputs"]

# uvicorn 실행
CMD ["uvicorn", "app:app", "--reload", "--host", "localhost", "--port", "9001"]
