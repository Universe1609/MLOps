FROM python:3.11
EXPOSE 5000/tcp

WORKDIR /app

COPY requirements.txt .
COPY models/modelos1 /models/modelo1
COPY models/modelos2 models/modelo2
COPY styles /app/styles

ENV IN_DOCKER_CONTAINER Yes

RUN echo pip --version

RUN pip intall --upgrade pip

RUN pip3 install -r requirements.txt

COPY app.py .

CMD [ "python", "./app.py" ]

