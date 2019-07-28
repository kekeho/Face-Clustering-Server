FROM python:3.7-stretch

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
