FROM continuumio/anaconda3:latest

WORKDIR /api

COPY requirements.txt /api/requirements.txt

RUN conda install faiss-cpu=1.3.0 -c pytorch -y

COPY . /api

