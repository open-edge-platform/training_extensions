# Contributing to Geti Edge 

**Table of Contents**

- [Contributing to Geti Edge ](#contributing-to-geti-edge)
  - [Configuration](#configuration)
  - [Setting up a local development](#setting-up-a-local-development)
    - [Local client & server with uv and npm](#local-client--server-with-uvhttpsdocsastralshuv-and-npmhttpswwwnpmjscom)
      - [Local server](#local-server)
      - [Local client](#local-client)
        - [Testing the UI](#testing-the-ui)
    - [Docker](#docker)
      - [With MQTT](#with-mqtt)

## Configuration

To start out with Geti Edge development make sure your `data` folder contains all files that are referenced to from your [`config.yaml`](./config.yaml) file.

Assuming your `config.yaml` is,
```
source:
  source_type: video_file  # [webcam, ip_camera, video_file, images_folder]
  video_path: data/media/video.mp4
sinks:
  - destination_type: folder  # [folder, mqtt, dds, ros]
    folder_path: data/output
    output_formats:
      - image_original
      - image_with_predictions
      - predictions
    rate_limit: 0.2  # output every 5 seconds
```

Then your `data/media` folder should contain a `video.mp4` file.

## Setting up a local development

### Local client & server with [uv](https://docs.astral.sh/uv/) and [npm](https://www.npmjs.com/)

First make sure you have uv and npm installed,
- [Installing uv](https://docs.astral.sh/uv/getting-started/installation/#installing-a-package)
- [Installing npm](https://nodejs.org/en/download)

#### Local server

First install the server dependencies,
```shell
uv sync
```

then run,
```shell
uv run fastapi run --port 7860
```

or start fastapi server in development mode,
```shell
uv run fastapi dev --port 7860
```

#### Local client

```
cd ui/
npm install
```

```
npm run start
```

##### Testing the UI

Use the following commands to test your changes.

```
npm run format
npm run type-check
npm run test:unit
npm run test:component
```

### Docker

If you don't have `uv` or `npm` installed then the quickest way to get started is using our docker compose setup.
```shell
docker compose up
```

This will first build our images and then start a server and client that will be available at http://geti-edge.localhost

If you want to develop for Geti Edge without installing uv or npm locally then you can also use the dev compose file,
```shell
docker compose -f docker-compose.dev.yaml up --watch
```

#### With MQTT

Use the `--profile mqtt` option to enable a local mqtt broker.

```shell
docker compose -f docker-compose.dev.yaml --profile mqtt up --watch
```

This also comes with a [MQTT Client UI](https://mqttx.app/web) that you can use to test your MQTT integrations, this is available at http://geti-edge-mqtt.localhost.


