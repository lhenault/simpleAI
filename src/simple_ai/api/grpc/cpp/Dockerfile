FROM ubuntu:22.04

# Update, install
RUN apt update && \
    apt install -y autoconf libtool pkg-config wget build-essential git

# Create user instead of using root
# RUN groupadd -r user && useradd -r -g user user
# USER user

# Define workdir
ENV APPDIR=/app
WORKDIR $APPDIR

# Env setup
ENV GRPC_INSTALL_DIR=$WORKDIR/.local
RUN mkdir -p $GRPC_INSTALL_DIR

# Update CMake
RUN wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-x86_64.sh
RUN sh cmake-linux.sh -- --skip-license --prefix=$GRPC_INSTALL_DIR

# Copy files
COPY . .

# Install (see: https://grpc.io/docs/languages/cpp/quickstart/#install-grpc)
ENV PATH="$GRPC_INSTALL_DIR/bin:$PATH"
RUN git clone --recurse-submodules -b v1.52.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc
WORKDIR $APPDIR/grpc
RUN mkdir -p cmake/build
# Explanation : https://stackoverflow.com/questions/51027077/possible-to-use-pushd-popd-in-dockerfile
RUN bash -xc "\
    pushd cmake/build; \
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$GRPC_INSTALL_DIR ../..; \
    make -j 4; \
    make install; \
    popd; \
    "

# Test installation with HelloWorld app:
ENV APPPATH=grpc/examples/cpp/helloworld
WORKDIR $APPDIR/$APPPATH
RUN mkdir -p cmake/build 
RUN bash -xc "\
    pushd cmake/build; \
    cmake -DCMAKE_PREFIX_PATH=$GRPC_INSTALL_DIR ../.. ; \
    make -j 4 ; \
    popd; \
    "

# Entry point, CMD
ENTRYPOINT ["/app/grpc/examples/cpp/helloworld/cmake/build/greeter_server"]
