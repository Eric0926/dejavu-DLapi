FROM continuumio/anaconda3:latest

WORKDIR /api

COPY requirements.txt /api/requirements.txt

RUN python3 -m pip3 install --no-cache-dir -r requirements.txt

COPY . /api

