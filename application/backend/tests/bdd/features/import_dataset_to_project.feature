# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Import Dataset To Project
  As a user of the application
  I want to import datasets to my projects
  So that I can add data to my projects from external sources

  @import_to_project @detection
  Scenario: Import detection dataset with label mapping
    Given A detection project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And a dataset with labels ["Chardonnay", "Blanc Fume", "Riesling"] exists
    And the dataset contains the following image distribution:
      | Label         | Training | Validation |
      | Chardonnay    | 5        | 2          |
      | Blanc Fume    | 5        | 0          |
      | Riesling      | 5        | 0          |
    And the dataset contains the following video frame distribution:
      | Label           | Training | Validation | Testing |
      | Chardonnay      | 2        | 1          | 1       |
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
      | Source Label | Target Label    |
      | Blanc Fume   | Sauvignon Blanc |
      | Riesling     | none            |
    Then the project statistics are:
      | Metric                 | Count |
      | images                 | 17    |
      | annotated_images       | 12    |
      | annotated_video_frames | 4     |
    And the project contains the following annotation instances:
      | Label           | Instances |
      | Chardonnay      | 11        |
      | Sauvignon Blanc | 5         |

  @import_to_project @classification
  Scenario: Import classification dataset with label mapping
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And a dataset with labels ["canine", "feline"] exists
    And the dataset contains the following image distribution:
      | Label  | Training |
      | canine | 3        |
      | feline | 3        |
    And the dataset contains the following video frame distribution:
      | Label  | Training | Validation | Testing |
      | canine | 3        | 1          | 1       |
      | feline | 3        | 1          | 1       |
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
      | Source Label | Target Label |
      | canine       | dog          |
      | feline       | cat          |
    Then the project statistics are:
      | Metric                 | Count |
      | images                 | 6     |
      | annotated_images       | 6     |
      | annotated_video_frames | 10    |
    And the project contains the following annotation instances:
      | Label | Instances |
      | cat   | 8         |
      | dog   | 8         |

  @import_to_project @multilabel
  Scenario: Import multilabel classification dataset
    Given A multilabel project "animals" with labels ["cat", "dog"] exists
    And a dataset with labels ["cat", "dog"] exists
    And the dataset contains the following image distribution:
      | Label | Training |
      | cat   | 3        |
      | dog   | 3        |
    And the dataset contains the following video frame distribution:
      | Label | Training | Validation | Testing |
      | cat   | 3        | 1          | 1       |
      | dog   | 3        | 1          | 1       |
    And the dataset is ready for import in staging directory
    When I import the dataset
    Then the project statistics are:
      | Metric                 | Count |
      | images                 | 6     |
      | annotated_images       | 6     |
      | annotated_video_frames | 10    |
    And the project contains the following annotation instances:
      | Label | Instances |
      | cat   | 8         |
      | dog   | 8         |

  @import_to_project @segmentation
  Scenario: Import segmentation dataset with label mapping
    Given A instance_segmentation project "traffic" with labels ["car", "person"] exists
    And a dataset with labels ["voiture", "personne"] exists
    And the dataset contains the following image distribution:
      | Label     | Training |
      | voiture   | 3        |
      | personne  | 3        |
    And the dataset contains the following video frame distribution:
      | Label    | Training | Validation | Testing |
      | voiture  | 1        | 1          | 1       |
      | personne | 1        | 1          | 1       |
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
      | Source Label | Target Label |
      | voiture      | none         |
      | personne     | none         |
    Then the project statistics are:
      | Metric                 | Count |
      | images                 | 6     |
      | annotated_images       | 0     |
      | annotated_video_frames | 0     |
    And the project contains the following annotation instances:
      | Label  | Instances |
      | car    | 0         |
      | person | 0         |
