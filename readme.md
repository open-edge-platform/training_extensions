# Geti Edge

Geti Edge is a no-code/low-code framework consisting of a series of task-centric reference implementations tailored for edge-based machine learning deployment.
Each reference implementation is purpose-built to address a specific machine learning domain, such as object detection, semantic segmentation, anomaly detection, and image captioning (VLM). 

## MVP Scope

The MVP scope for Geti Edge will be a minimal application that allows users to setup an inference pipeline by configuring an input source (focus on IP Camera or RTSP stream), upload a model and configure output hooks.
After configuring the pipeline the server runs inference on the input source and broadcasts data to its configured outputs. 
The `Input -> Inference -> Outputs[]`

### Configuration wizard

This wizard allows us to configure an inference pipeline.
We assume the user only has a single pipeline.

#### Input

The user can configure a single input source.

```
POST /api/pipeline/inputs

{
    id: Uuid;
    source: 'camera' | 'rtsp';
    address: string;
    port: string;
    // ... TBD
}
```

The first input that the user adds will be the default selected, optionally in the future we may decide to allow the user to add multiple inputs and have them (de)activate them

```
POST /api/pipeline/inputs/{:input_id}:activate

> This endpoint could be replaced by a PUT endpoint?
```

```
GET /api/pipeline/inputs
```

```
DELETE /api/pipeline/inputs/{:input_id}
```

```
PUT /api/pipeline/inputs/{:input_id}
```

#### Model

A user can upload t heir model by simply importing a zip file containing the model's `.xml` and `.bin` files (e.g. the downloaded OpenVINO model from Geti). 
Additionally the user can provide extra metadata such as the model name, version and original(?) model accuracy.
After importing the model the server should be able to determine the model's:
- Labels
- Size

> Note:
> The MVP will focus only on supporting OpenVINO models. At a later stage we can look into adding ONNX support or improting from a model deployment (Geti SDK, OVMS) file
> Another option is to download directly from Geti, but as edge scenarios are likely to be air gapped we should assume the application can not connect to their Geti instance.

```
POST /api/pipeline/models

{
    id: Uuid;
    name: string;
    version: number;
    file: Zip; // TBD
}
```

```
GET /api/pipeline/models

Array<{
    id: Uuid;
    is_active: boolean;
    name: string;
    version: number;
    labels: Array<Label>;
}>
```

```
DELETE /api/pipeline/models/{:model_id}
```

```
POST /api/pipeline/models/{:model_id}:activate
```

#### Outputs

After configuring their input source and selected model they can add outputs.
As an initial MVP I'd suggest to focus on either adding webhook or folder: folder is easy to test and verify while webhook allows us to implement the other destinations as an external service.

```
POST /api/pipeline/outputs

{
    destination: {
        type: 'mqtt' | 'dds' | 'webhook' | 'folder'
        // security headers TBD
    },
    as_json: boolean;
    as_image: boolean; // TBD?

}
```

```
POST /api/pipeline/outputs/{:output_id}:activate

> This endpoint could be replaced by a PUT endpoint?
```

```
DELETE /api/pipeline/outputs/{:input_id}
```

```
PUT /api/pipeline/outputs/{:input_id}
```

```
GET /api/pipeline/outputs/{:input_id}
```

### Live feed

Once the user has submitted their initial pipeline they can monitor it by opening Geti Edge's live feed.
The live feed is similar to Geti's annotator view, but instead of an interactive editor the user sees a the feed of their activated input source along with inference and monitoring data.

- The user/client needs to manually activate the WebRTC stream. If the stream isn't actively used the server will turn it off to safe compute
- By default monitoring data like model confidence and FPS is updated continuously: likely via webrtc / websockets.

#### Monitoring & Data collection

> TBD

### Build / edit pipeline

> TBD

### Example use case

A potato company wants to configure to perform quality control on potatoes moving over a belt. They've setup a single IP camera at the factory with a view on the belt.
They've trained a detection model with Geti that they want to use to count potatoes moving over their belt.
They've implemented a simple REST server that implements a webhook which takes prediction results as in input and stores their count.
To use Geti they:
- Select their IP camera as an input
- Upload their detection model 
- Select webhook as the output

After configuring the pipeline they open the Geti Edge feed to inspect if the model accurately detects potatoes from their camera.
After confirming that the model works as expected they close the application. The server continues to monitor the camera feed and sens data to their REST server.

