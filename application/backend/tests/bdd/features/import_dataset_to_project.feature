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
    And the dataset containing 5 training images labeled "Chardonnay"
    And the dataset containing 5 training images labeled "Blanc Fume"
    And the dataset containing 5 training images labeled "Riesling"
    And the dataset containing 2 validation images labeled "Chardonnay"
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
      | Source Label | Target Label    |
      | Blanc Fume   | Sauvignon Blanc |
      | Riesling     | none            |
    Then the project contains 7 annotated images labeled "Chardonnay"
    And the project contains 5 annotated images labeled "Sauvignon Blanc"
    And the project contains 5 unannotated images

  @import_to_project @classification
  Scenario: Import classification dataset with label mapping
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And a dataset with labels ["canine", "feline"] exists
    And the dataset containing 3 training images labeled "canine"
    And the dataset containing 3 training images labeled "feline"
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
      | Source Label | Target Label |
      | canine       | dog          |
      | feline       | cat          |
    Then the project contains 3 annotated images labeled "cat"
    And the project contains 3 annotated images labeled "dog"

  @import_to_project @multilabel
  Scenario: Import multilabel classification dataset
    Given A multilabel project "animals" with labels ["cat", "dog"] exists
    And a dataset with labels ["cat", "dog"] exists
    And the dataset containing 3 training images labeled "cat"
    And the dataset containing 3 training images labeled "dog"
    And the dataset is ready for import in staging directory
    When I import the dataset
    Then the project contains 3 annotated images labeled "cat"
    And the project contains 3 annotated images labeled "dog"

  @import_to_project @segmentation
  Scenario: Import segmentation dataset with label mapping
    Given A instance_segmentation project "traffic" with labels ["car", "person"] exists
    And a dataset with labels ["voiture", "personne"] exists
    And the dataset containing 3 training images labeled "voiture"
    And the dataset containing 3 training images labeled "personne"
    And the dataset is ready for import in staging directory
    When I import the dataset with label mappings:
        | Source Label | Target Label |
        | voiture      | none         |
        | personne     | none         |
    Then the project contains 6 unannotated images
