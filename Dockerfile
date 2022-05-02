FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN apt update && \
    apt install -y git gcc ffmpeg libsm6 libxext6 libgl1-nvidia-glx:i386

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

RUN git clone https://github.com/open-mmlab/mmdetection.git

RUN cd mmdetection && pip install -e .

RUN pip install Pillow==7.0.0

ENV SHELL /bin/bash
