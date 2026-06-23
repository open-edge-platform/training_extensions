# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Import Dataset As New Project
  As a user of the application
  I want to import datasets as new projects
  So that I can create new projects from external datasets

  @import_as_new_project @detection
  Scenario Outline: Import dataset as a new detection project
    Given A <dataset_type> dataset with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the dataset contains the following image distribution:
      | Label           | Training | Validation |
      | Chardonnay      | 5        | 2          |
      | Sauvignon Blanc | 5        | 2          |
      | Cabernet Franc  | 5        | 2          |
    And the dataset is ready for import in staging directory
    When I import the dataset as a new detection project with name "grapes" and labels ["Chardonnay", "Sauvignon Blanc"]
    Then the project "grapes" is created with labels ["Chardonnay", "Sauvignon Blanc"]
    And the project statistics are:
      | Metric                 | Count |
      | images                 | 21    |
      | annotated_images       | 14    |
      | annotated_video_frames | 0     |
    And the project contains the following annotation instances:
      | Label           | Instances |
      | Chardonnay      | 7         |
      | Sauvignon Blanc | 7         |

    Examples:
        | dataset_type          |
        | detection             |
        | instance_segmentation |

  @import_as_new_project @classification
  Scenario: Import classification dataset as a new project
    Given A classification dataset with labels ["cat", "dog"] exists
    And the dataset contains the following image distribution:
      | Label | Training | Validation | Testing |
      | cat   | 3        | 1          | 1       |
      | dog   | 3        | 1          | 1       |
    And the dataset is ready for import in staging directory
    When I import the dataset as a new classification project with name "animals" and labels ["cat"]
    Then the project "animals" is created with labels ["cat"]
    And the project statistics are:
      | Metric                 | Count |
      | images                 | 10    |
      | annotated_images       | 5     |
      | annotated_video_frames | 0     |
    And the project contains the following annotation instances:
      | Label | Instances |
      | cat   | 5         |

  @import_as_new_project @multilabel
  Scenario Outline: Import dataset as a new multilabel project
    Given A <dataset_type> dataset with labels ["cat", "dog"] exists
    And the dataset contains the following image distribution:
      | Label | Training |
      | cat   | 3        |
      | dog   | 3        |
    And the dataset is ready for import in staging directory
    When I import the dataset as a new multilabel project with name "animals" and labels ["cat", "dog"]
    Then the project "animals" is created with labels ["cat", "dog"]
    And the project statistics are:
      | Metric                 | Count |
      | images                 | 6     |
      | annotated_images       | 6     |
      | annotated_video_frames | 0     |
    And the project contains the following annotation instances:
      | Label | Instances |
      | cat   | 3         |
      | dog   | 3         |

    Examples:
        | dataset_type          |
        | multilabel            |
        | detection             |
        | instance_segmentation |

  @import_as_new_project @segmentation
  Scenario Outline: Import dataset as a new instance_segmentation project
    Given A <dataset_type> dataset with labels ["car", "person"] exists
    And the dataset contains the following image distribution:
      | Label  | Training | Testing |
      | car    | 3        | 3       |
      | person | 3        | 3       |
    And the dataset is ready for import in staging directory
    When I import the dataset as a new instance_segmentation project with name "traffic" and labels ["car", "person"]
    Then the project "traffic" is created with labels ["car", "person"]
    And the project statistics are:
      | Metric                 | Count |
      | images                 | 12    |
      | annotated_images       | 12    |
      | annotated_video_frames | 0     |
    And the project contains the following annotation instances:
      | Label  | Instances |
      | car    | 6         |
      | person | 6         |

    Examples:
        | dataset_type          |
        | detection             |
        | instance_segmentation |
