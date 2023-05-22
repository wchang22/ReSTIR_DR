FROM ubuntu:20.04

RUN apt-get update -yy
RUN apt-get install -yy \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  gnupg \
  wget

RUN apt-get install -yy clang-9 libc++-9-dev libc++abi-9-dev cmake ninja-build
RUN apt-get install -yy libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev
RUN apt-get install -yy python3-dev python3-distutils python3-setuptools
RUN apt-get install -yy jupyter

RUN ln -sf /bin/python3.8 /usr/bin/python

RUN mkdir -p /tmp
WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py

RUN pip install ipython matplotlib numpy

RUN ln -sf /usr/bin/clang-9 /usr/bin/clang
RUN ln -sf /usr/bin/clang++-9 /usr/bin/clang++

ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++
ENV SHELL /bin/bash

WORKDIR /root
ENTRYPOINT bash
