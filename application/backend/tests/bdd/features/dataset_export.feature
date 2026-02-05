Feature: Dataset Export
  As a user of the application
  I want to export datasets in various formats
  So that I can use the data in other applications

  Scenario Outline: Export detection project dataset
    Given A "detection" project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the project dataset has 10 images with annotations in subset "training"
    And the project dataset has 2 images with annotations in subset "validation"
    And the project dataset has 2 images with annotations in subset "testing"
    And the project dataset has 2 unannotated images
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset has <expected images> images

    Examples:
     | export format | archive name     | filters                                | expected images |
#     | YOLO          | dataset-yolo.zip | { "include_unannotated": true }        | 16              |  FIXME:
     | YOLO          | dataset-yolo.zip | { "subsets": ["training", "testing"] } | 12              |
     | YOLO          | dataset-yolo.zip | { "include_unannotated": false }       | 14              |
#     | DATUMARO_V2   | dataset.zip      | { "include_unannotated": true }        | 16              |  FIXME:
     | DATUMARO_V2   | dataset.zip      | { "subsets": ["training", "testing"] } | 12              |
     | DATUMARO_V2   | dataset.zip      | { "include_unannotated": false }       | 14              |
#     | COCO          | TODO: uncomment when COCO export is fixed
