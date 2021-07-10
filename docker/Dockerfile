# getting base image
FROM daskdev/dask-notebook



# RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --no-install-recommends && \
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --no-install-recommends && \
  apt-get install -y --no-install-recommends locales && \
  locale-gen en_US.UTF-8 && \
  apt-get install -y --no-install-recommends software-properties-common && \
  apt-get install -y --no-install-recommends \
        make \
        automake \
        libpq-dev \
        libffi-dev \
        gfortran \
        g++ \
        git \
        libboost-program-options-dev \
        libtool \
        libxrender1 \
        wget \
        ca-certificates \
        python3-pyqt5 \
        pyqt5-dev-tools \
        qttools5-dev-tools \
        libgl1-mesa-glx \
        libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
        curl && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
#RUN jupyter labextension install dask-labextension

CMD ["echo","Hello world"]
WORKDIR /home/jovyan/project

