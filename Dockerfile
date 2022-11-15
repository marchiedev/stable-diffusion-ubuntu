FROM python:3.9.7

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y \
    libgl1 libglib2.0-0

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

#download modal
RUN python3 demo.py --num-inference-steps 1 --prompt "test" --output /tmp/test.jpg

EXPOSE 80

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]