# Task and labels

A _task_ represents an instance of a machine learning problem (classification, detection, etc.) with a specific
set of _labels_, i.e. the classes or categories that can be assigned to objects in a dataset.
For example, a task to classify images of animals could have labels such as "cat", "dog", and "bird".
Depending on the task definition, labels can be mutually exclusive or not. For example, in a task of classifying
photos of landscapes, one image could have the labels "mountain" and "lake" without any contradiction. Conversely,
when classifying vehicles by type, an object can be a "car" or a "motorcycle", but not both at the same time.

In Geti Tune, every _project_ addresses a specific _task_. The currently supported task types are:

- **Classification**: Categorize images into one (_multiclass_) or more (_multilabel_) classes.
- **Detection**: Identify and localize objects in images using bounding boxes.
- **Instance Segmentation**: Identify and segment objects in images using polygons.

After creating a project, the task type is fixed and cannot be changed. On the other hand, the user can
customize the label structure of the task by adding, removing or editing labels.

Labels are uniquely identified by a UUID, which is immutable. They also have a name and other frontend-related
attributes (color, hotkey, etc.) that can be modified by the user.

## Storage

### DB schema

```
,-----------------------------------.
|projects                           |
|-----------------------------------|
|id : UUID                          |
|...                                |
|task_type : TEXT NOT NULL          |
|exclusive_labels : BOOLEAN NOT NULL|
`-----------------------------------'
                  | 1
                  |
                  | 1..*
 ,---------------------------------.
 |labels                           |
 |---------------------------------|
 |id : UUID                        |
 |name : TEXT NOT NULL             |
 |project_id : UUID NOT NULL       |
 |created_at : TIMESTAMP NOT NULL  |
 |updated_at : TIMESTAMP NOT NULL  |
 |color : TEXT                     |
 |hotkey : TEXT                    |
 `---------------------------------'
```

In the database, the information about the task is stored in two tables: `projects` and `labels`:

- `projects` contains details about the project including the task type and whether the labels are mutually exclusive.
- `labels` contains the individual labels, each associated with a task. Each label has its own id, name
  and other attributes for UX purposes (color, hotkey, etc.).

#### Example

Here is an example of how the database tables could look like for a task that classifies chess pieces by type.
The field `exclusive_labels` is set to `true` because each piece is strictly one of the six types, you can't have
an item that is both a "Pawn" and a "Knight".

_'projects' table_

| id                                   | task_type      | exclusive_labels |
| ------------------------------------ | -------------- | ---------------- |
| 550e8400-e29b-41d4-a716-446655440000 | classification | true             |

_'labels' table_

| id                                   | project_id                           | name   | color   | hotkey |
| ------------------------------------ | ------------------------------------ | ------ | ------- | ------ |
| 550e8400-e29b-41d4-a716-446655440002 | 550e8400-e29b-41d4-a716-446655440000 | Pawn   | #A0A0A0 | p      |
| 550e8400-e29b-41d4-a716-446655440003 | 550e8400-e29b-41d4-a716-446655440000 | Knight | #8090C0 | n      |
| 550e8400-e29b-41d4-a716-446655440004 | 550e8400-e29b-41d4-a716-446655440000 | Bishop | #A070C0 | b      |
| 550e8400-e29b-41d4-a716-446655440005 | 550e8400-e29b-41d4-a716-446655440000 | Rook   | #C07070 | r      |
| 550e8400-e29b-41d4-a716-446655440006 | 550e8400-e29b-41d4-a716-446655440000 | Queen  | #D090D0 | q      |
| 550e8400-e29b-41d4-a716-446655440007 | 550e8400-e29b-41d4-a716-446655440000 | King   | #E0C070 | k      |

_Note: the columns `created_at` and `updated_at` are omitted for simplicity._

## REST API

See the [API reference](api.md) for details.

## Design choices

This section provides a breakdown of some of the design choices made in the task and labels management system.

### Supported task types

The first iteration of Geti Tune only support three of the most common task types: classification, detection and
segmentation. This simplification comes from Geti's experience, where the proliferation of task types, each with its
own quirks, has led to a complex and hard-to-maintain codebase. Geti Tune is open to extending the supported task types
in the future, but only after careful estimation of the value in relation to the complexity. Note that other products
in the Geti ecosystem, such as Geti Inspect, offer additional tasks like anomaly detection.

### Project - task association

Projects are created with a specific task type, and it cannot be changed later.
This constraint is in place to ensure that the project pipeline, when active, always outputs data with a consistent type
(e.g. bounding boxes or polygons, but not a mix of both), thus allowing for stronger assumptions about the dataset
annotations and model input/outputs.

### Task - labels association

The task defines one or more labels - the minimum number actually depends on the task type.
Labels strictly belong to the task (project): they can't be reparented to a different project, but they can be modified
or deleted. Labels can't exist independently of the project because the type of task affects certain semantic
propositions about the labels themselves.
For example, an empty label cannot exist in a multiclass classification task, while it can in detection or
multilabel classification. Other examples of tasks with peculiar label structures or behaviors are anomaly detection,
hierarchical classification and keypoint detection: while these tasks are not yet supported, the design aims to be
extensible enough to accommodate them in the future.
