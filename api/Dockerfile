FROM continuumio/anaconda3:5.2.0

WORKDIR /api

COPY requirements.txt /api/requirements.txt

RUN conda install faiss-cpu -c pytorch -y

RUN python3 -m pip install wrapt --upgrade --ignore-installed

RUN python3 -m pip install -r requirements.txt

COPY . /api

