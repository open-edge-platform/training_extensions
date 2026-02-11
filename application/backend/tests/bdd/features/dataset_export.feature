Feature: Dataset Export
  As a user of the application
  I want to export datasets in various formats
  So that I can use the data in other applications

  @export @detection
  Scenario Outline: Export detection project dataset
    Given A "detection" project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the project dataset has 10 images with annotations in subset "training"
    And the project dataset has 2 images with annotations in subset "validation"
    And the project dataset has 2 images with annotations in subset "testing"
    And the project dataset has 2 unannotated images in subset "training"
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the exported dataset has <expected images> images

    Examples:
     | export format | archive name     | filters                                | expected images |
     | YOLO          | dataset-yolo.zip | { "include_unannotated": true }        | 16              |
     | YOLO          | dataset-yolo.zip | { "subsets": ["training", "testing"] } | 14              |
#     | YOLO          | dataset-yolo.zip | { "include_unannotated": false }       | 14              |
     | GETI          | dataset-geti.zip | { "include_unannotated": true }        | 16              |
     | GETI          | dataset-geti.zip | { "subsets": ["training", "testing"] } | 14              |
#     | GETI          | dataset-geti.zip | { "include_unannotated": false }       | 14              |
     | COCO          | dataset-coco.zip | { "include_unannotated": true }        | 16              |
     | COCO          | dataset-coco.zip | { "subsets": ["training", "testing"] } | 14              |
#     | COCO          | dataset-coco.zip | { "include_unannotated": false }       | 14              |

#  @export @classification
#  Scenario Outline: Export classification project dataset
#    Given A "classification" project "animals" with labels ["cat", "dog"] exists
#    And the project dataset has 5 images with annotations in subset "training"
#    And the project dataset has 5 images with annotations in subset "validation"
#    And the project dataset has 5 images with annotations in subset "testing"
#    And the project dataset has 5 unannotated images in subset "training"
#    When I export the project dataset in <export format> format with filters=<filters>
#    Then the staged dataset archive <archive name> should exist
#    And the exported dataset has <expected images> images
#
#    Examples:
#     | export format | archive name  | filters                                             | expected images |
#     | GETI   | dataset-geti.zip     | { "include_unannotated": true }                     | 20              |
#     | GETI   | dataset-geti.zip     | { "subsets": ["training", "testing"] }              | 15              |
#     | GETI   | dataset-geti.zip     | { "labels": ["cat"], "include_unannotated": false } | 9               |
#     | GETI   | dataset-geti.zip     | { "include_unannotated": false }                    | 15              |
##     | VOC    | dataset-voc.zip      | { "include_unannotated": true }        | 20              |
##     | VOC    | dataset-voc.zip      | { "subsets": ["training", "testing"] } | 10              |
##     | VOC    | dataset-voc.zip      | { "include_unannotated": false }       | 15              |

#  @export @segmentation
#  Scenario Outline: Export segmentation project dataset
#    Given An "instance_segmentation" project "traffic" with labels ["car", "person"] exists
#    And the project dataset has 6 images with annotations in subset "training"
#    And the project dataset has 3 images with annotations in subset "validation"
#    And the project dataset has 3 images with annotations in subset "testing"
#    And the project dataset has 3 unannotated images
#    When I export the project dataset in <export format> format with filters=<filters>
#    Then the staged dataset archive <archive name> should exist
#    And the exported dataset has <expected images> images
#
#    Examples:
#     | export format | archive name  | filters                                    | expected images |
##     | GETI   | dataset-geti.zip     | { "include_unannotated": true }            | 15              |
#     | GETI   | dataset-geti.zip     | { "subsets": ["training", "validation"] }  | 9               |
#     | GETI   | dataset-geti.zip     | { "include_unannotated": false }           | 12              |
##     | COCO   | dataset-coco.zip     | { "include_unannotated": true }            | 15              |
##     | COCO    | dataset-voc.zip      | { "subsets": ["training", "validation"] } | 9               |
##     | COCO    | dataset-voc.zip      | { "include_unannotated": false }          | 12              |