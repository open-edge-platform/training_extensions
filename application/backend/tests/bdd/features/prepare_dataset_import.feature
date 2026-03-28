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
    And the project contains the following video frame distribution:
      | Label           | Training | Validation | Testing |
      | Chardonnay      | 2        | 1          | 1       |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <image count> images
    And the staged dataset with name=dataset has <frame count> video frames

    Examples:
      | export format | image count | frame count |
      | YOLO          | 49          | 0           |
      | GETI          | 45          | 4           |
      | COCO          | 49          | 0           |

  @prepare @classification
  Scenario Outline: Prepare classification dataset for import
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | cat   | 10       | 3          | 2       |
      | dog   | 10       | 3          | 2       |
    And the project contains the following video frame distribution:
      | Label | Training | Validation | Testing |
      | cat   | 3        | 1          | 1       |
      | dog   | 3        | 1          | 1       |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <image count> images
    And the staged dataset with name=dataset has <frame count> video frames

    Examples:
      | export format | image count | frame count |
      | GETI          | 30          | 10          |
      | VOC           | 40          | 0           |

  @prepare @segmentation
  Scenario Outline: Prepare segmentation dataset for import
    Given An instance_segmentation project "traffic" with labels ["car", "person"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | car   | 10       | 3          | 2       |
      | person| 10       | 3          | 2       |
    And the project contains the following video frame distribution:
      | Label  | Training | Validation | Testing |
      | car    | 1        | 1          | 1       |
      | person | 1        | 1          | 1       |
    And the project dataset is exported in <export format> format
    When I prepare the staged dataset archive for import
    Then the staged dataset is ready for import
    And the staged dataset with name=dataset has <image count> images
    And the staged dataset with name=dataset has <frame count> video frames

    Examples:
      | export format | image count | frame count |
      | COCO          | 36          | 0           |
      | GETI          | 30          | 6           |
