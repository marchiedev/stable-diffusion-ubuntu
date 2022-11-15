FROM python:3.9.7

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

EXPOSE 80

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5001"]