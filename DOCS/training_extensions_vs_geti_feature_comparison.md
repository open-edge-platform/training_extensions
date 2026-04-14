# training_extensions application vs Geti feature comparison

## Scope and method

- `training_extensions` in this document means the current application surface under `application/backend` and `application/ui`.
- I also checked the OTX library under `library/src/otx` to separate "repo-level capability exists" from "application currently exposes it".
- `geti` in this document is based on local source under `C:\git_repos\geti`, especially `interactive_ai`, `platform`, `web_ui`, and REST schema files.
- This comparison is source-backed first, with docs used only as supporting context.

## 1. Supported project types

| Area | training_extensions application | Geti |
| --- | --- | --- |
| Project structure | Single-task project. `Project` has one `task`, and `TaskType` only exposes `classification`, `detection`, `instance_segmentation`. | Multi-task project graph. `Project` owns a `TaskGraph`, exposes ordered `tasks`, and supports chained task configurations. |
| Classification | Supported. Current app supports classification with `exclusive_labels=True/False`, which maps to multiclass vs multilabel flows. | Supported. REST schemas also include `classification_hierarchical`. |
| Object detection | Supported. | Supported. |
| Instance segmentation | Supported. | Supported. |
| Semantic segmentation | Not exposed by current app task enum or app model manifests. | Supported. |
| Rotated detection | Not exposed by current app task enum or app model manifests. | Supported. |
| Keypoint detection | Not exposed by current app task enum or app model manifests. | Supported. |
| Anomaly projects | Not exposed by current app task enum or app model manifests. | Supported. |
| Task-chained project types | Not supported in current app. | Supported, including `detection_classification` and `detection_segmentation`. |

### Notes

- `training_extensions` app evidence:
  - `application/backend/app/models/project.py`
  - `application/backend/app/models/task.py`
  - `application/backend/app/datumaro_converter/__init__.py`
- `Geti` evidence:
  - `C:\git_repos\geti\interactive_ai\libs\iai_core_py\iai_core\entities\project.py`
  - `C:\git_repos\geti\interactive_ai\libs\iai_core_py\iai_core\entities\task_graph.py`
  - `C:\git_repos\geti\interactive_ai\services\api\schemas\dataset_import_export\project_types.yaml`
  - `C:\git_repos\geti\interactive_ai\services\api\schemas\configuration\requests\task_chain_configuration.yaml`

## 2. Supported model types

| Task family | training_extensions application manifests | Geti manifests |
| --- | --- | --- |
| Classification | `deit_tiny`, `dinov2`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2_s`, `mobilenet_v3_large` | `deit_tiny`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2_l`, `efficientnet_v2_s`, `mobilenet_v3_large`, `mobilenet_v3_small` |
| Detection | `atss_mobilenet_v2`, `dfine_l/m/x`, `dinov3_detr_s/m/l`, `rfdetr_s/m/l`, `rtdetr_50`, `ssd_mobilenet_v2`, `yolox_t/s/l/x` | `atss_mobilenetv2`, `atss_resnext101`, `dfine_x`, `rtdetr_18/50/101`, `rtmdet_tiny`, `ssd_mobilenetv2`, `yolox_t/s/l/x` |
| Instance segmentation | `maskrcnn_efficientnet_b2`, `maskrcnn_r50`, `maskrcnn_swin_t`, `rfdetr_s/m/l/xl`, `rtmdet_tiny` | `maskrcnn_efficientnetb2b`, `maskrcnn_r50_v1`, `maskrcnn_r50_v2`, `maskrcnn_swin_t`, `rtmdet_tiny` |
| Semantic segmentation | Not exposed by current app. | `dinov2_s`, `litehrnet_18`, `litehrnet_s`, `litehrnet_x`, `segnext_t/s/b` |
| Rotated detection | Not exposed by current app. | `maskrcnn_efficientnetb2b`, `maskrcnn_r50_v1`, `maskrcnn_r50_v2` |
| Keypoint detection | Not exposed by current app. | `rtmpose` |
| Anomaly | Not exposed by current app. | `padim`, `stfpm`, `uflow` |
| Visual prompting | Not exposed by current app. | `visual_prompting_model` |

### Important repo-level caveat

The broader `training_extensions` repo already contains OTX recipe directories for:

- `classification`
- `detection`
- `instance_segmentation`
- `semantic_segmentation`
- `rotated_detection`
- `keypoint_detection`

That means the repo-level ML stack is ahead of the current application surface. The app currently exposes only three project/task families, while the OTX library already contains recipes and test assets for more.

### Notes

- `training_extensions` app manifests:
  - `application/backend/app/supported_models/manifests/classification`
  - `application/backend/app/supported_models/manifests/detection`
  - `application/backend/app/supported_models/manifests/instance_segmentation`
- `training_extensions` repo-level OTX evidence:
  - `library/src/otx/recipe`
  - `library/tests/assets`
- `Geti` model evidence:
  - `C:\git_repos\geti\interactive_ai\supported_models\geti_supported_models\manifests`
  - `C:\git_repos\geti\interactive_ai\supported_models\geti_supported_models\default_models.py`

## 3. Feature matrix

| Feature area | training_extensions application | Geti |
| --- | --- | --- |
| Architecture style | Local application with backend/API/UI, local DB schema, filesystem-backed artifacts, and in-process job queue. | Distributed platform: `platform` services, `interactive_ai` services, workflow jobs, web UI, and Kubernetes deployment support. |
| Workspaces / organizations | Not found in current app API, data models, or DB schema. | Supported. Dedicated workspace and organization models, CRUD services, and REST/API documentation exist. |
| Multi-user / RBAC | Not found in current app routes or schema. | Supported. User, membership, role, invitation, and workspace-role flows are implemented. |
| Personal access tokens | Not found. | Supported. Personal access token model and REST/API endpoints exist. |
| Authentication-aware API surface | Not found beyond source/sink credential handling. | Supported. Many APIs are scoped by organization/workspace/project and use session/auth dependencies. |
| Dataset management | Supported. Media upload, dataset item listing/filtering, subset assignment, dataset statistics, dataset revisions. | Supported. Also includes dataset storages and richer platform workflows. |
| Image/video annotation | Supported. Image annotations, video frame annotations, rectangle/polygon/full-image shapes, prediction-assisted review state. | Supported. Image, video-frame, and video-range annotation endpoints, annotation templates, workspace-scoped APIs. |
| Dataset import/export | Supported. Prepare/import/export jobs and label mapping/filtering for dataset transfer. | Supported. |
| Project import/export | Not found. | Supported via dedicated service and workflows. |
| Model revisions / lineage | Supported. Per-project model revisions with parent-child lineage and dataset revision linkage. | Supported. |
| Model variants | Supported. Variant records for `pytorch`, `openvino`, `onnx`, with `fp16`, `fp32`, `int8`. | Supported, plus deployment-oriented registration graphs and packaging. |
| Quantization / optimization | Supported via dedicated quantization job and INT8 model variants. | Supported via optimize workflow and deployment flows. |
| Training metrics / logs | Supported. Training metrics, logs, SSE status/log streaming, download APIs. | Supported. Also backed by separate jobs/workflow infrastructure. |
| Job orchestration | In-process queue; router explicitly notes job details are not persisted across server restarts. | Dedicated jobs service and Flyte workflows for train, optimize, model test, dataset IE, and project IE. |
| Real-time / batch inference | Supported. Batch media prediction plus per-project pipeline entity with source/sink/model selection. | Supported. Predictions, pipelines, inference gateway, and deployment flows exist. |
| Pipeline sources/sinks | Supported. USB camera, IP camera, video file, images folder; folder, MQTT, ROS, webhook sinks; WebRTC endpoints. | Platform has prediction/pipeline/deployment APIs, but not the same local app source/sink abstraction as the current training_extensions app. |
| Active learning | Not found. | Supported. Dedicated active learning endpoints, entities, samplers, and storage repos exist. |
| Visual prompting / one-shot prompting | Not found. | Supported. Dedicated visual prompting service uses SAM-based models and stores learned reference features. |
| Deployment package generation | Not found. App can download model binaries, but not package OVMS/Geti SDK deployment bundles. | Supported. Deployment package endpoints generate `ovms` and `geti_sdk` packages. |
| Code deployment | Not found. | Supported by dedicated code deployment endpoints and resource-management code. |

## 4. Practical summary

The current `training_extensions` application is best described as a local, project-centric training and inference workbench:

- strong for local dataset management, annotation, training, model revisioning, quantization, and source/sink-based inference
- limited to three app-exposed task families
- no source evidence for workspaces, multi-user collaboration, RBAC, PATs, project chaining, active learning, visual prompting, or deployment packaging

Geti is a broader collaborative platform:

- wider project/task coverage, including chained projects and additional CV domains
- multi-workspace, multi-user, token-based, role-aware platform services
- dedicated active learning, visual prompting, project import/export, and deployment-package workflows

## 5. Key evidence paths

### training_extensions

- `application/backend/app/models/project.py`
- `application/backend/app/models/task.py`
- `application/backend/app/models/model_revision.py`
- `application/backend/app/models/pipeline.py`
- `application/backend/app/models/source.py`
- `application/backend/app/models/sink.py`
- `application/backend/app/db/schema.py`
- `application/backend/app/api/routers/projects.py`
- `application/backend/app/api/routers/media.py`
- `application/backend/app/api/routers/datasets.py`
- `application/backend/app/api/routers/dataset_revisions.py`
- `application/backend/app/api/routers/jobs.py`
- `application/backend/app/api/routers/models.py`
- `application/backend/app/api/routers/system.py`
- `application/backend/app/api/routers/webrtc.py`
- `application/backend/app/supported_models/manifests`
- `library/src/otx/recipe`
- `library/tests/assets`

### Geti

- `C:\git_repos\geti\interactive_ai\libs\iai_core_py\iai_core\entities\project.py`
- `C:\git_repos\geti\interactive_ai\libs\iai_core_py\iai_core\entities\task_graph.py`
- `C:\git_repos\geti\interactive_ai\services\api\schemas\dataset_import_export\project_types.yaml`
- `C:\git_repos\geti\interactive_ai\services\api\schemas\projects\requests\post\task.yaml`
- `C:\git_repos\geti\interactive_ai\services\api\schemas\configuration\requests\task_chain_configuration.yaml`
- `C:\git_repos\geti\interactive_ai\supported_models\geti_supported_models\manifests`
- `C:\git_repos\geti\interactive_ai\supported_models\geti_supported_models\default_models.py`
- `C:\git_repos\geti\interactive_ai\services\director\app\communication\endpoints\active_learning_endpoints.py`
- `C:\git_repos\geti\interactive_ai\services\director\app\active_learning\usecases\active_set_retrieval_usecase.py`
- `C:\git_repos\geti\interactive_ai\services\visual_prompt\app\services\visual_prompt_service.py`
- `C:\git_repos\geti\interactive_ai\services\resource\app\communication\rest_endpoints\annotation_endpoints.py`
- `C:\git_repos\geti\interactive_ai\services\resource\app\communication\rest_endpoints\code_deployment_endpoints.py`
- `C:\git_repos\geti\interactive_ai\services\project_ie\app\usecases\project_upload_usecase.py`
- `C:\git_repos\geti\interactive_ai\workflows\project_ie\job\workflows\export_project_workflow.py`
- `C:\git_repos\geti\platform\services\account\app\models\workspace.go`
- `C:\git_repos\geti\platform\services\account\app\grpc\workspace\workspace.go`
- `C:\git_repos\geti\platform\services\account\app\models\personal_access_token.go`
- `C:\git_repos\geti\platform\services\user_directory\app\endpoints\user_management\invite_user.py`
