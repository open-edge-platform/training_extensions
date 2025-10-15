# Pipeline management

In Geti Tune, the _inference pipeline_ (or _pipeline_ for short) is the central concept that represents the entire
processing and inference flow of a video stream from input sources to output sinks.
Each project has its own pipeline, which can be thought of as an "engine" that manages and executes a sequence
of processing steps, ensuring data flows through them properly.

![Diagram showing the sequence of processing stages](media/pipeline-stages.jpg)

The simplest, most minimal pipeline consists of three elements:

- A _source_, namely the origin of the stream. This could be, for example, a camera or a video file.
- A _model_, which elaborates the input data to generate predictions.
- A _sink_, that is where the results should be dispatched. This could be, for example, a folder or a message queue.

In practice, there are usually other stages in between that analyze or transform the data stream:

- Monitoring of the input data, to detect potential issues with the source (e.g. lens blockage, lighting changes, ...)
- Monitoring of the model predictions, to detect potential issues with the model (e.g. concept drift, out-of-distribution samples, ...)
- Data collection, to save interesting frames for later inspection or to improve the model via fine-tuning.

In Geti Tune, the configuration of sources and sinks is decoupled from the configuration of the pipeline.
In other words, sources and sinks are configured independently, then the pipeline connects to the selected ones.
This abstraction not only makes it easy to switch from one source to another (e.g. two or more cameras),
but also allows the reuse of these pre-configured components in other projects and pipelines.

Pipeline stages operate at the frame level. To facilitate the integration of the processing stages and the transfer
of data from one to another, all stages adhere to a unified format for input and output data (`StreamData` entity).
This also allows for easy extension of the pipeline with new intermediate processing steps, without affecting the
existing stages.

A pipeline can be in one of two states: _running_ or _idle_. A running pipeline constantly tries to load frames,
process them and output to the sink; conversely, an idle pipeline does not perform any work until activated.
Only one pipeline can be active at a given time: this limitation is due to the hardware resources required to run
the model inference, which are usually not sufficient to run multiple pipelines simultaneously at full speed.

![Diagram showing pipeline components](media/pipeline-components.jpg)

## Entities

### Sources

Supported sources include:

- Cameras (webcams, industrial cameras, IP cameras, ...)
- Files (videos, images)

The configuration of a source consists of a _type_ and a set of source-specific attributes. For example:

```yaml
type: video_file
video_path: videos/sports/golf.mp4
```

or

```yaml
type: ip_camera
stream_url: rtsp://192.168.1.100:554/stream
```

Each source is identified by a unique id, and optionally a friendly name chosen by the user (e.g. "Room 3 camera 4").

### Sinks

Supported sinks include:

- Local filesystem (folder)
- Webhooks
- Messaging frameworks (MQTT, ROS2, ...)

The configuration of a sink consists of a _type_ and a set of sink-specific attributes. For example:

```yaml
type: mqtt
broker_host: localhost
broker_port: 1883
topic: predictions
```

Each sink is identified by a unique id, and optionally a friendly name chosen by the user (e.g. "Robot 3").

### Models

A pipeline performs inference using models in OpenVINO IR format (.xml + .bin), converted after fine-tuning.
The supported task types include image classification, object detection and instance segmentation.

A model is identified by a unique id.

### Pipelines

A pipeline is a combination of a source, a model and a sink. These components can be replaced as needed.

The configuration of a pipeline looks like the following:

```yaml
source: <source_id>
model: <model_id>
sink: <sink_id>
```

A pipeline is identified by a unique id, and optionally a friendly name chosen by the user (e.g. "Production").

Users can export a pipeline to a zip file, so that they can later import it to another environment.
The exported archive includes a full YAML definition of the source and sink. The user may decide to include the model
or not: if included, then the archive also contains the model binaries.

The status of a pipeline can be 'running' or 'idle'. A running pipeline constantly tries to load frames, process them
and output to the sink; conversely, an idle pipeline does not perform any work until activated. Multiple pipelines
may be running at the same time, provided that they don't share an _exclusive_ component: for example, two pipelines
configured to read from the same camera cannot run together. Exclusivity is a property of the source/sink type: e.g.
a physical camera requires exclusive access, whereas an IP camera or a topic does not.

## REST API

See the [API reference](api.md).
