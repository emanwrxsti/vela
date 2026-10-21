# supr-meow-vela — Vela PoI v2 GPU miner. Multi-stage: build from source, ship a slim runtime.
ARG CUDA_IMAGE=13.0.0
FROM nvidia/cuda:${CUDA_IMAGE}-devel-ubuntu24.04 AS build
ARG CUDA_ARCHS="80;86;89;90;100;120"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt /src/
COPY src/           /src/src/
COPY vendor/        /src/vendor/
# OpenSSL supplies TLS and CPU-dispatched SHA-256.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
 && cmake --build build -j"$(nproc)" --target supr-meow-vela \
 && strip build/supr-meow-vela

FROM nvidia/cuda:${CUDA_IMAGE}-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libssl3 openssh-server \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/supr-meow-vela /app/
# Bundle the exact CUDA runtime libs the binary links (cudart + cuBLAS/cuBLASLt).
RUN mkdir -p /app/lib
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13   /app/lib/
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13   /app/lib/
COPY --from=build /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13 /app/lib/
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY README.md LICENSE NOTICE /app/
COPY licenses/ /app/licenses/
RUN chmod +x /app/entrypoint.sh /app/supr-meow-vela
ENV LD_LIBRARY_PATH=/app/lib
RUN mkdir -p /models
VOLUME ["/models"]
EXPOSE 22
ENTRYPOINT ["/app/entrypoint.sh"]
