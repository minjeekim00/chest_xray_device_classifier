FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive

RUN apt install -y build-essential software-properties-common ca-certificates vim \
                 wget git zlib1g-dev nasm cmake

RUN apt-get install -y libgl1-mesa-glx
RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 setuptools==59.5.0
RUN pip install SimpleITK opencv-python natsort easydict notebook


#WORKDIR /workspace
#COPY . ${WORKDIR}

#WORKDIR /workspace/stylegan3

#RUN conda env create -f environment.yml
#RUN conda init

