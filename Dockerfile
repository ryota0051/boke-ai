FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /work

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt update -y && apt install -y git libopencv-dev

COPY ./requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install git+https://github.com/rinnakk/japanese-clip.git@v0.2.0

RUN python -m unidic download
