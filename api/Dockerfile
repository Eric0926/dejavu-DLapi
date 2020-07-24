FROM continuumio/anaconda3:latest

WORKDIR /api

COPY requirements.txt /api/requirements.txt

RUN conda install --file requirements.txt

COPY . /api

