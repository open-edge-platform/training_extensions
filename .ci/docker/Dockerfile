#########################################################
## Python Environment with CUDA
#########################################################

FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04@sha256:c1869c30f46fff478a37ed58d9dace7e08519541274f03424d0b78bd35b2c73a AS python_base_cuda
LABEL maintainer="OpenVINO Training Extensions Development Team"

ARG uid
ARG gid

ENV DEBIAN_FRONTEND="noninteractive"

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y \
        software-properties-common \
        wget \
        ffmpeg \
        curl \
        nodejs \
        unzip \
        git \
        cron \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set /tmp folder cleaning schedule at 7PM every day which is older than a day
RUN echo "0 19 * * * find /tmp/* -mtime +1 -exec rm -rf {} \;" >> ./cron_clean_tmp.txt && \
    crontab cron_clean_tmp.txt && \
    # Create a non-root user with having given UID & GID
    groupadd -r -g ${gid} validation && \
    useradd -l -r -m validation -g ${gid} -u ${uid} && \
    echo "${gid}:${uid}"

COPY SectigoPublicServerAuthenticationCAOVR36-base64.crt /usr/local/share/ca-certificates/
COPY SectigoPublicServerAuthenticationRootR46-base64.crt /usr/local/share/ca-certificates/
COPY SectigoRSAOrganizationValidationSecureServerCA-base64.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates --fresh

USER validation

WORKDIR /home/validation


#########################################################
## OTX Development Env
#########################################################

FROM python_base_cuda as getitune_development_env

RUN mkdir actions-runner
WORKDIR /home/validation/actions-runner
ARG ACTIONS_RUNNER_VER=2.333.1
# download actions-runner and extract it
RUN curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v${ACTIONS_RUNNER_VER}/actions-runner-linux-x64-${ACTIONS_RUNNER_VER}.tar.gz && \
    tar xzf ./actions-runner-linux-x64.tar.gz && \
    rm actions-runner-linux-x64.tar.gz
