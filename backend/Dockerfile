LABEL api.covid-detector.backend

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt-get -y update \
        && apt-get install -y --no-install-recommends \ 
        git gcc ffmpeg libsm6 libxext6

RUN pip install -U openmim

RUN mim install mmcv-full

RUN git clone https://github.com/open-mmlab/mmdetection.git

RUN cd mmdetection && pip install -e .

WORKDIR /app/backend

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "run.py"]
