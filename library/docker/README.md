# How to build cuda and cuda-pretrained-ready Docker images

1. By executing the following commands, it will build two Docker images: `getitune:${GETITUNE_VERSION}-cuda` and `getitune:${GETITUNE_VERSION}-cuda-pretrained-ready`.

   ```console
   git clone https://github.com/open-edge-platform/training_extensions.git
   cd docker
   ./build.sh
   ```

2. After that, you can check whether the images are built correctly such as

   ```console
   docker image ls | grep getitune
   ```

   Example:

   ```console
   otx                                           2.0.0-cuda-pretrained-ready                    4f3b5f98f97c   3 minutes ago   14.5GB
   otx                                           2.0.0-cuda                                     8d14caccb29a   8 minutes ago   10.4GB
   ```

`getitune:${GETITUNE_VERSION}-cuda` is a minimal Docker image where getitune is installed with CUDA supports. On the other hand, `getitune:${GETITUNE_VERSION}-cuda-pretrained-ready` includes all the model pre-trained weights that getitune provides in addition to `getitune:${GETITUNE_VERSION}-cuda`.
