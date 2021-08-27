FROM ubuntu:16.04

#ENV http_proxy=http://10.41.249.28:8080 https_proxy=http://10.41.249.28:8080

RUN apt-get update -yq && apt-get install -yq build-essential cmake git pkg-config wget zip
RUN apt-get install -yq libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
RUN apt-get install -yq libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -yq libgtk2.0-dev
RUN apt-get install -yq libatlas-base-dev gfortran
RUN apt-get install -yq software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -yq python3.7
RUN apt-get install -yq python3.7 python3-dev python3-pip python3-setuptools python3-tk git
RUN apt-get remove -yq python-pip python3-pip && wget https://bootstrap.pypa.io/get-pip.py && python3.7 get-pip.py
RUN python3.7 -m pip install numpy
RUN cd ~ && git clone https://github.com/Itseez/opencv.git
RUN cd ~/opencv && mkdir build && cd build
# cd'ing again for debugging docker layers
RUN cd ~/opencv/build && cmake . -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_opencv_python3=yes -D PYTHON_EXECUTABLE=/usr/bin/python3 ..
RUN cd ~/opencv/build && make -j8
RUN cd ~/opencv/build && make install
RUN rm -rf /root/opencv/
RUN mkdir -p /root/tf-openpose
RUN rm -rf /tmp/*.tar.gz
RUN apt-get clean && rm -rf /tmp/* /var/tmp* /var/lib/apt/lists/*
RUN rm -f /etc/ssh/ssh_host_* && rm -rf /usr/share/man?? /usr/share/man/??_*

COPY . /root/tf-openpose/
WORKDIR /root/tf-openpose/

RUN cd /root/tf-openpose/ && python3.7 -m pip install -U setuptools
# pycocotools gives an error
RUN python3.7 -m pip install pycocotools
RUN python3.7 -m pip install -r requirements.txt
RUN python3.7 -m pip install tensorflow

RUN cd /root && git clone https://github.com/cocodataset/cocoapi
RUN python3.7 -m pip install cython
RUN cd cocoapi/PythonAPI && python3 setup.py build_ext --inplace && python3 setup.py build_ext install
RUN mkdir /coco && cd /coco && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
RUN unzip annotations_trainval2017.zip && rm -rf annotations_trainval2017.zip

ENTRYPOINT ["python3", "pose_dataworker.py"]

#ENV http_proxy= https_proxy=
