FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /work

RUN apt update -y && apt install git -y

COPY ./requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install git+https://github.com/rinnakk/japanese-clip.git@v0.2.0
