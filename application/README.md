<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->

<div align="center">

<img src="../assets/geti-header.png" alt="Geti™ - Build and deploy computer vision AI models with minimal effort and data">

**Full-stack web application to build and deploy computer vision AI models, powered by the [getitune](../library) library.**

[![python](https://img.shields.io/badge/python-3.13-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2026.2-purple)]()

[Quick start](#quick-start) •
[Docs](#documentation) •
[License](#license)

</div>

## Quick start

There are several ways to run Geti, choose the method that best fits your workflow:

- **Docker (recommended)** [[instructions]](#run-with-docker) - download and run one of the pre-built Docker images, or build one yourself
- **MSIX App (Windows)** [[instructions]](#install-as-windows-app) - install as a desktop application
- **Run from Source (for development)** [[instructions]](#run-from-source-for-development) - run the server and the UI as standalone components

### Run with Docker

The easiest and most portable way to run Geti is through Docker.
We provide pre-built images for Intel® XPU and NVIDIA® CUDA platforms, or you can build your own image from source.

> [!WARNING]
> The official Docker images for Geti 3.x have not been released publicly yet.
> The only way to run Geti at the moment is to build the image from source (see below).

<details>
<summary><strong>Prerequisites</strong></summary>

On the host system:

- Docker v29+ [[docs]](https://docs.docker.com/)
- (Optional, recommended) Just v1.46+ [[docs]](https://github.com/casey/just)
- (Only for Intel® XPU) the latest driver suitable with your HW [[docs]](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[docs]](https://www.nvidia.com/Download/index.aspx)

</details>

Install the pre-built image of your choice:

```bash
# Recommended choice if you have a modern Intel® CPU or GPU
docker pull ghcr.io/open-edge-platform/geti-xpu

# Alternative: support for NVIDIA CUDA-enabled platforms
docker pull ghcr.io/open-edge-platform/geti-cuda

# Alternative: lightweight CPU-only image
docker pull ghcr.io/open-edge-platform/geti-cpu
```

<details>
<summary><strong>Advanced: Build the image</strong></summary>

Geti Docker images can be built from source using the [`Dockerfile`](./docker/Dockerfile) in the `application` directory.
This can be useful if you want to customize the application or optimize it for a specific usecase that is not covered by the pre-built images.

The instructions below use `just` to simplify the build process, but you can also build the image manually with `docker build` if you prefer.

From the `application` directory:

```bash
# Choice of accelerator: 'xpu', 'cuda', or 'cpu'
just build-image --accelerator xpu
```

Run `just --usage build-image` to see all available build options.

</details>

Once you have downloaded or built the Geti image, use the `run-image` command to launch the application.

```bash
# Run the image build with support for a specific accelerator ('xpu', 'cuda', or 'cpu')
just run-image --accelerator xpu --port 8080
```

Run `just --usage run-image` to see all available runtime options.

After the container starts, you can access the Geti web application at [**http://localhost:8080**](http://localhost:7860) (assuming default settings).

<details>
<summary><strong>Advanced: Run with a TURN server</strong></summary>

Geti uses WebRTC for the real-time inference streaming visualization in the UI. WebRTC requires the browser to establish
a direct connection to the backend's media server: when the application is deployed in a restricted network environment,
such as behind a corporate firewall or NAT, the connection may fail due to the inability to traverse the network boundaries.
To address this issue, you can set up a TURN server that acts as a relay between the browser and the backend,
allowing the WebRTC traffic to pass through the restricted network.

First, set up a TURN server by running `just run-coturn` from the `application` directory.
This command will launch a Coturn TURN server in a Docker container with default credentials.

Next, pass the `--coturn` option when launching the Geti container via `just run-image`. That's it, now you should be
able to view the predictions in the 'Inference' page of the UI.

Later, when you no longer need the TURN server, stop it with `just stop-coturn`.

</details>

<details>
<summary><strong>Advanced: Browse the app storage</strong></summary>

The Geti application uses a Docker volume named `geti-data` to persistently store all datasets, models, and other objects.
You can browse the contents of this volume by running a temporary container that mounts the volume and lists the files.

```shell
# List the contents of the root directory in the `geti-data` volume
docker run --rm -v geti-data:/data alpine ls -l /data

# List the model files of a specific project (replace <PROJECT_ID> with the actual ID)
docker run --rm -v geti-data:/data alpine ls -l /data/projects/<PROJECT_ID>/models

# List the media files of a specific project (replace <PROJECT_ID> with the actual ID)
docker run --rm -v geti-data:/data alpine ls -l /data/projects/<PROJECT_ID>/dataset
```

</details>

<details>
<summary><strong>Troubleshooting: View the logs</strong></summary>

When running Geti with Docker, all logs are stored in the `geti-logs` Docker volume.
You can view these logs by running a temporary container that mounts the volume and prints the log files to the console.

**Application logs:**

```bash
# Print the logs of the application container to the console
docker run --rm -v geti-logs:/logs alpine cat /logs/app.log | jq -r '.text'

# Or save the logs to a file for easier browsing
docker run --rm -v geti-logs:/logs alpine cat /logs/app.log | jq -r '.text' > geti-logs.txt
```

**Job logs:**

```bash
# List the available job logs
docker run --rm -v geti-logs:/logs alpine ls -l /logs/jobs

# Print the logs of a specific job to the console
docker run --rm -v geti-logs:/logs alpine cat /logs/jobs/<job_id>.log | jq -r '.text'
```

**Logs of other worker processes:**

```bash
# Print the logs of the inference pipeline stream loader
docker run --rm -v geti-logs:/logs alpine cat /logs/workers/streamloader.log | jq -r '.text'

# Print the logs of the inference worker
docker run --rm -v geti-logs:/logs alpine cat /logs/workers/inference.log | jq -r '.text'
```

</details>

### Install as Windows app

> [!WARNING]
> The MSIX App for Geti 3.0 has not been released yet.

### Run from source (for development)

For development purposes, you can run the Geti server and UI as standalone components without Docker.

<details>
<summary><strong>Prerequisites</strong></summary>

- Just v1.46+ [[docs]](https://github.com/casey/just)
- (Only for Intel® XPU) the latest driver suitable with your HW [[docs]](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[docs]](https://www.nvidia.com/Download/index.aspx)
- Node.js v24.2+ [[docs]](https://nodejs.org/)

</details>

<details>
<summary><strong>Run the server</strong></summary>

To run the server, use the `run-server` command after initializing the environment with `venv`:

```bash
# From the repo root
cd application/backend

# Initialize the environment with the appropriate accelerator support (cpu, xpu, or cuda)
just venv --accelerator xpu

# Run the server
just run-server
```

Run `just --usage run-server` for a full list of options for running the server. Notably, by passing the option
`--setup-demo`, the application will be pre-populated with demo data, including sample datasets and pre-trained models.

</details>

<details>
<summary><strong>Run the UI</strong></summary>

After running the server, build and launch the UI in a separate terminal:

```bash
# From the repo root
cd application/ui

# Install dependencies and build
npm install
npm run build

# Start the UI
npm run start
```

After the UI starts, you can access the Geti web application at [**http://localhost:3000**](http://localhost:3000) (assuming default settings).

</details>

## Documentation

Please check the [documentation website](https://docs.geti.intel.com/) for detailed guides, API reference,
and other resources to help you get the most out of Geti.

<details>
<summary><strong>Advanced: generate the API spec from source </strong></summary>

The OpenAPI specification for the Geti REST API can be generated with the `generate-api-spec` command:

```bash
# From the repo root
cd application/backend

# Generate the OpenAPI spec and save it to a custom location
just gen-api-spec --output-path="openapi.json"
```

</details>

## License

The Geti source code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The Windows MSIX App is licensed under the [Intel Simplified Software License](https://software.intel.com/sites/landingpage/pintool/intel-simplified-software-license.txt).
For more information, refer to the [LICENSE](../LICENSE) page.
