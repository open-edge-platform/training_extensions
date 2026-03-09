# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Dataset Export
  As a user of the application
  I want to export datasets in various formats
  So that I can use the data in other applications

  @export @detection
  Scenario Outline: Export detection project dataset
    Given A detection project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the project contains the following image distribution:
      | Label           | Training | Validation | Testing |
      | Chardonnay      | 6        | 2          | 2       |
      | Sauvignon Blanc | 6        | 2          | 2       |
      | Cabernet Franc  | 6        | 2          | 2       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <expected images> images

    Examples:
      | export format | archive name     | filters                                                    | expected images |
      | YOLO          | dataset-yolo.zip | { }                                                        | 30              |
      | YOLO          | dataset-yolo.zip | { "subsets": ["training", "testing"] }                     | 24              |
      | YOLO          | dataset-yolo.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 10              |
      | GETI          | dataset-geti.zip | { }                                                        | 30              |
      | GETI          | dataset-geti.zip | { "subsets": ["training", "testing"] }                     | 24              |
      | GETI          | dataset-geti.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 10              |
      | COCO          | dataset-coco.zip | { }                                                        | 30              |
      | COCO          | dataset-coco.zip | { "subsets": ["training", "testing"] }                     | 24              |
      | COCO          | dataset-coco.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 10              |

  @export @classification
  Scenario Outline: Export classification project dataset
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | cat   | 5        | 5          | 5       |
      | dog   | 5        | 5          | 5       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <expected images> images

    Examples:
      | export format | archive name     | filters                                             | expected images |
      | GETI          | dataset-geti.zip | { }                                                 | 30              |
      | GETI          | dataset-geti.zip | { "subsets": ["training", "testing"] }              | 20              |
      | GETI          | dataset-geti.zip | { "labels": ["cat"], "include_unannotated": false } | 15              |
#     | VOC    | dataset-voc.zip      | { }                                                 | 30              |
#     | VOC    | dataset-voc.zip      | { "subsets": ["training", "testing"] }              | 20              |
#     | VOC    | dataset-voc.zip      | { "labels": ["cat"], "include_unannotated": false } | 15              |

  @export @segmentation
  Scenario Outline: Export segmentation project dataset
    Given An instance_segmentation project "traffic" with labels ["car", "person"] exists
    And the project contains the following image distribution:
      | Label  | Training | Validation | Testing |
      | car    | 7        | 5          | 3       |
      | person | 7        | 5          | 3       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <expected images> images

    Examples:
      | export format | archive name     | filters                       | expected images |
      | GETI          | dataset-geti.zip | { }                           | 30              |
      | GETI          | dataset-geti.zip | { "subsets": ["validation"] } | 10              |
      | GETI          | dataset-geti.zip | { "labels": ["person"] }      | 30              |
      | COCO          | dataset-coco.zip | { }                           | 30              |
      | COCO          | dataset-coco.zip | { "subsets": ["validation"] } | 10              |
      | COCO          | dataset-coco.zip | { "labels": ["person"] }      | 30              |
