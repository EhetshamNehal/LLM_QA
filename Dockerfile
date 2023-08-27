FROM python:3.10

WORKDIR /llama2

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3","model.py"," --host=0.0.0.0"]