FROM python:3.7-buster

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev 

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
COPY ./keras-yolo3/ ./keras-yolo3/
COPY ./slim_yolo.py ./keras-yolo3/
COPY ./yolo.h5 ./keras-yolo3/model_data/
COPY ./app.py ./
COPY ./app.cfg ./
COPY ./imgs ./imgs
RUN rm -Rf .test
RUN mkdir -p /tmp

EXPOSE 5000
CMD ["python", "app.py"]

