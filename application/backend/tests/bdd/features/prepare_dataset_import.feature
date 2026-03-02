# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Prepare Dataset For Import
  As a user of the application
  I want to prepare datasets for import
  So that I can validate and stage data before adding it to my projects

  @prepare @detection
  Scenario Outline: Prepare detection dataset for import
    Given A detection project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the project contains the following image distribution:
      | Label           | Training | Validation |
      | Chardonnay      | 10       | 5          |
      | Sauvignon Blanc | 10       | 5          |
      | Cabernet Franc  | 10       | 5          |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <expected images> images

    Examples:
     | export format | expected images |
     | YOLO          | 45              |
     | GETI          | 45              |
     | COCO          | 45              |

  @prepare @classification
  Scenario Outline: Prepare classification dataset for import
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | cat   | 10       | 3          | 2       |
      | dog   | 10       | 3          | 2       |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <expected images> images

    Examples:
     | export format | expected images |
     | GETI          | 30              |
#     | VOC           | 30              |

  @prepare @segmentation
  Scenario Outline: Prepare segmentation dataset for import
    Given An instance_segmentation project "traffic" with labels ["car", "person"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | car   | 10       | 3          | 2       |
      | person| 10       | 3          | 2       |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <expected images> images

    Examples:
     | export format | expected images |
     | COCO          | 30              |
     | GETI          | 30              |
