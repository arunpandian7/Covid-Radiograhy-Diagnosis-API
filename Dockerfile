FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-get -y update \
        && apt-get install -y --no-install-recommends \ 
        git gcc ffmpeg libsm6 libxext6

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html

RUN git clone https://github.com/open-mmlab/mmdetection.git

RUN cd mmdetection && pip install -e .

RUN pip install Pillow==7.0.0

ENV SHELL /bin/bash
