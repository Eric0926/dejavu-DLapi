FROM continuumio/anaconda3:5.2.0

WORKDIR /api

ENV LD_PRELOAD=/opt/conda/lib/libmkl_def.so:/opt/conda/lib/libmkl_avx.so:/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_intel_lp64.so:/opt/conda/lib/libmkl_intel_thread.so:/opt/conda/lib/libiomp5.so

COPY requirements.txt /api/requirements.txt

RUN conda install faiss-cpu -c pytorch -y

RUN python3 -m pip install wrapt --upgrade --ignore-installed

RUN python3 -m pip install -r requirements.txt

RUN export LD_PRELOAD=/opt/conda/lib/libmkl_core.so:/opt/conda/lib/libmkl_sequential.so

COPY . /api

