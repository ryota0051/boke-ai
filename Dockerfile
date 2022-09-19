FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /work

COPY ./requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt
