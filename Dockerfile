FROM python:3.8-slim-buster

RUN apt update -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]
