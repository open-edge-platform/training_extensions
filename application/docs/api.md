# REST API

## Input / output

### Sources

| Method   | Path                       | Payload       | Return          | Description                       |
| -------- | -------------------------- | ------------- | --------------- | --------------------------------- |
| `POST`   | `/api/sources`             | source config | source id       | Create and configure a new source |
| `GET`    | `/api/sources`             | -             | list of sources | List the available sources        |
| `GET`    | `/api/sources/<id>`        | -             | source info     | Get info about a source           |
| `PATCH`  | `/api/sources/<id>`        | source config | -               | Reconfigure an existing source    |
| `POST`   | `/api/sources/<id>:export` | -             | yaml file       | Export a source to file           |
| `POST`   | `/api/sources:import`      | yaml file     | source id       | Import a source from file         |
| `DELETE` | `/api/sources/<id>`        | -             | -               | Remove a source                   |

### Sinks

| Method   | Path                     | Payload     | Return        | Description                     |
| -------- | ------------------------ | ----------- | ------------- | ------------------------------- |
| `POST`   | `/api/sinks`             | sink config | sink id       | Create and configure a new sink |
| `GET`    | `/api/sinks`             | -           | list of sinks | List the available sinks        |
| `GET`    | `/api/sinks/<id>`        | -           | sink info     | Get info about a sink           |
| `PATCH`  | `/api/sinks/<id>`        | sink config | -             | Reconfigure an existing sink    |
| `POST`   | `/api/sinks/<id>:export` | -           | yaml file     | Export a sink to file           |
| `POST`   | `/api/sinks:import`      | yaml file   | sink id       | Import a sink from file         |
| `DELETE` | `/api/sinks/<id>`        | -           | -             | Remove a sink                   |

## Projects

| Method   | Path                                        | Payload            | Return           | Description                       |
| -------- | ------------------------------------------- | ------------------ | ---------------- | --------------------------------- |
| `POST`   | `/api/projects`                             | name, task, labels | project info     | Create a new project              |
| `GET`    | `/api/projects`                             | -                  | list of projects | List the available projects       |
| `GET`    | `/api/projects/<id>`                        | -                  | project info     | Get info about a project          |
| `PATCH`  | `/api/projects/<id>`                        | name               | project info     | Rename a project                  |
| `DELETE` | `/api/projects/<id>`                        | -                  | -                | Delete a project                  |
| `PATCH`  | `/api/projects/<id>/labels`                 | labels to change   | task and labels  | Add, remove or edit labels        |
| `GET`    | `/api/projects/<id>/training_configuration` | -                  | training config  | Get the training configuration    |
| `PATCH`  | `/api/projects/<id>/training_configuration` | training config    | -                | Update the training configuration |

### Pipelines

| Method  | Path                                  | Payload                    | Return        | Description                           |
| ------- | ------------------------------------- | -------------------------- | ------------- | ------------------------------------- |
| `GET`   | `/api/projects/<id>/pipeline`         | -                          | pipeline info | Get info about a project's pipeline   |
| `PATCH` | `/api/projects/<id>/pipeline`         | ids of source, sink, model | pipeline info | Reconfigure the project's pipeline    |
| `POST`  | `/api/projects/<id>/pipeline:enable`  | -                          | pipeline info | Activate a project's pipeline         |
| `POST`  | `/api/projects/<id>/pipeline:disable` | -                          | pipeline info | Deactivate a project's pipeline       |
| `POST`  | `/api/projects/<id>/pipeline:capture` | -                          | -             | Collect the next frame to the dataset |

#### Inference metrics

| Method | Path                                  | Payload | Return       | Description                                      |
| ------ | ------------------------------------- | ------- | ------------ | ------------------------------------------------ |
| `GET`  | `/api/projects/<id>/pipeline/metrics` | -       | metrics info | Get inference metrics (latency, throughput, ...) |

## Datasets

| Method   | Path                                              | Payload | Return                | Description                                        |
| -------- | ------------------------------------------------- | ------- | --------------------- | -------------------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset/items`                | -       | list of dataset items | List the dataset items (option 'with_annotations') |
| `GET`    | `/api/projects/<id>/dataset/items/<id>`           | -       | dataset item info     | Get info about a dataset item                      |
| `GET`    | `/api/projects/<id>/dataset/items/<id>/binary`    | -       | binary                | Get the image data of a dataset item (full res)    |
| `GET`    | `/api/projects/<id>/dataset/items/<id>/thumbnail` | -       | binary                | Get the thumbnail of a dataset item                |
| `POST`   | `/api/projects/<id>/dataset/items`                | binary  | media info            | Upload an image to the dataset                     |
| `DELETE` | `/api/projects/<id>/dataset/items/<id>`           | -       | -                     | Delete a dataset item                              |

### Annotations

| Method   | Path                                                | Payload         | Return          | Description                               |
| -------- | --------------------------------------------------- | --------------- | --------------- | ----------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset/items/<id>/annotations` | -               | annotation info | Get the annotation/prediction for a media |
| `POST`   | `/api/projects/<id>/dataset/items/<id>/annotations` | annotation info | annotation info | Annotate a media                          |
| `DELETE` | `/api/projects/<id>/dataset/items/<id>/annotations` | -               | -               | Delete the annotation for a media         |

### Tags

| Method  | Path                                         | Payload                   | Return       | Description                                 |
| ------- | -------------------------------------------- | ------------------------- | ------------ | ------------------------------------------- |
| `GET`   | `/api/projects/<id>/dataset/items/<id>/tags` | -                         | list of tags | List the tags of a dataset item             |
| `GET`   | `/api/projects/<id>/dataset/tags`            | -                         | list of tags | List the tags used in the dataset           |
| `PATCH` | `/api/projects/<id>/dataset/items/tags`      | items, tags to add/remove | -            | Apply or remove tags from one or more items |

### Views

| Method   | Path                                          | Payload             | Return        | Description                      |
| -------- | --------------------------------------------- | ------------------- | ------------- | -------------------------------- |
| `POST`   | `/api/projects/<id>/dataset/views`            | name                | view info     | Create a new dataset view        |
| `GET`    | `/api/projects/<id>/dataset/views`            | -                   | list of views | List the dataset views           |
| `GET`    | `/api/projects/<id>/dataset/views/<id>`       | -                   | view info     | Get info about a dataset view    |
| `GET`    | `/api/projects/<id>/dataset/views/<id>/items` | -                   | list of items | List the items in a dataset view |
| `POST`   | `/api/projects/<id>/dataset/views/<id>/items` | items ids or filter | -             | Add items to a dataset view      |
| `DELETE` | `/api/projects/<id>/dataset/views/<id>/items` | items ids or filter | -             | Remove items from a dataset view |
| `DELETE` | `/api/projects/<id>/dataset/views/<id>`       | -                   | -             | Delete a dataset view            |

### Models

| Method   | Path                                          | Payload | Return         | Description                            |
| -------- | --------------------------------------------- | ------- | -------------- | -------------------------------------- |
| `GET`    | `/api/projects/<id>/models`                   | -       | list of models | List all the models in a project       |
| `GET`    | `/api/projects/<id>/models/<model_id>`        | -       | model info     | Get info about a specific model        |
| `GET`    | `/api/projects/<id>/models/<model_id>/labels` | -       | labels         | Get the labels used to train the model |
| `DELETE` | `/api/projects/<id>/models/<model_id>`        | -       | -              | Delete a model (option 'weights_only') |

### Dataset revisions (training datasets, etc...)

| Method   | Path                                                        | Payload | Return        | Description                                        |
| -------- | ----------------------------------------------------------- | ------- | ------------- | -------------------------------------------------- |
| `GET`    | `/api/projects/<id>/dataset_revisions/items`                | -       | list of items | List the dataset items (option 'with_annotations') |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>`           | -       | item info     | Get info about a dataset item                      |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>/binary`    | -       | binary        | Get the image data of a dataset item (full res)    |
| `GET`    | `/api/projects/<id>/dataset_revisions/items/<id>/thumbnail` | -       | binary        | Get the thumbnail of a dataset item                |
| `DELETE` | `/api/projects/<id>/dataset_revisions`                      | -       | -             | Remove the dataset files to free space             |

## Jobs

| Method | Path                    | Payload              | Return       | Description                                        |
| ------ | ----------------------- | -------------------- | ------------ | -------------------------------------------------- |
| `POST` | `/api/jobs`             | job type and params  | job id       | Create and submit a new job                        |
| `GET`  | `/api/jobs`             | -                    | list of jobs | List the jobs in a project (scheduled or running)  |
| `GET`  | `/api/jobs/<id>`        | -                    | job info     | Get info about a specific job                      |
| `POST` | `/api/jobs/<id>:cancel` | -                    | -            | Cancel a job                                       |
| `GET`  | `/api/jobs/<id>/status` | -                    | job status   | Stream real-time status updates for a specific job |
| `GET`  | `/api/jobs/<id>/logs`   | -                    | job logs     | Stream real-time log output for a specific job     |

Job types:

- `train`
- `prepare_dataset_for_import`
- `import_dataset_to_existing_project`
- `import_dataset_as_new_project`
- `export_dataset`
- `stage_dataset`

