# Use Ubuntu as the base image
FROM ubuntu:20.04

# Copy the files from your project
WORKDIR /app
# Update the package manager
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tzdata 

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
         make \
         cmake \
         wget \
         unzip \
         vim \
         git \
         libopencv-dev \
         libboost-all-dev 

RUN apt-get install -y build-essential
RUN apt-get update && apt-get -y install cmake protobuf-compiler

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip -O libtorch.zip
RUN unzip -o libtorch.zip
ENV LIBTORCH /app/libtorch
ENV LD_LIBRARY_PATH /app/libtorch/lib:$LD_LIBRARY_PATH


# Install the necessary dependencies

COPY ["CMakeLists.txt","image_classification.cpp","label.txt","resnet18.pt","./"] 
            
# Change the working directory


# Build the project
RUN mkdir build && cd build && cmake .. -DCMAKE_PREFIX_PATH=$PWD/../libtorch .. && make 

RUN cd build 

CMD ["./image_classification"]