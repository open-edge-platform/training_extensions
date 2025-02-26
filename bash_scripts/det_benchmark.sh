#!/bin/bash
DATASET_PREFIX="/home/yuchunli/datasets"
OUTPUT_PATH="/home/yuchunli/otx-2.x/otx-workspace/det-benchmark"

RTDETR_50="/home/yuchunli/otx-2.x/src/otx/recipe/detection/rtdetr_50.yaml"
YOLOX_TINY="/home/yuchunli/otx-2.x/src/otx/recipe/detection/yolox_tiny.yaml"
YOLOX_S="/home/yuchunli/otx-2.x/src/otx/recipe/detection/yolox_s.yaml"
YOLOX_L="/home/yuchunli/otx-2.x/src/otx/recipe/detection/yolox_l.yaml"
YOLOX_X="/home/yuchunli/otx-2.x/src/otx/recipe/detection/yolox_x.yaml"
DFINE_X="/home/yuchunli/otx-2.x/src/otx/recipe/detection/dfine_x.yaml"

AEROMONAS="Vitens-Aeromonas-coco"
COLIFORM="Vitens-Coliform-coco"
CHICKEN="Chicken-Real-Time-coco-roboflow"
SKIN="skindetect-roboflow"
WGISD="wgisd-coco"
BLUEBERRY="BlueBerry23.v1i.coco-mmdetection"
CARPART="car-seg.v1i.coco-mmdetection"
PACKAGE="factory_package.v1i.coco-mmdetection"
FASHION="fashion-categories-coco-roboflow"
CITYSCAPES="cityscapes_cut"
COCO_CAR_PERSON="coco_car_person_medium"

MODELS=(${RTDETR_50} ${YOLOX_TINY} ${YOLOX_S} ${YOLOX_L} ${YOLOX_X} ${DFINE_X})
DATASET_ARRAY=(${COLIFORM} ${AEROMONAS} ${CARPART} ${PACKAGE} ${CHICKEN} ${SKIN} ${WGISD} ${BLUEBERRY} ${FASHION} ${CITYSCAPES} ${COCO_CAR_PERSON})

for model in ${MODELS[@]}; do
    for dataset in ${DATASET_ARRAY[@]}; do
        if [ $model = ${RTDETR_50} ]
        then
            FOLDER="RTDETR_50"
        elif [ $model = ${YOLOX_TINY} ]
        then
            FOLDER="YOLOX_TINY"
        elif [ $model = ${YOLOX_S} ]
        then
            FOLDER="YOLOX_S"
        elif [ $model = ${YOLOX_L} ]
        then
            FOLDER="YOLOX_L"
        elif [ $model = ${YOLOX_X} ]
        then
            FOLDER="YOLOX_X"
        else
            FOLDER="DFINE_X"
        fi

        mkdir -p ${OUTPUT_PATH}/${FOLDER}/$dataset \

        otx train --config "$model" \
            --data_root "${DATASET_PREFIX}/$dataset" \
            --engine.device "gpu" \
            --work_dir "${OUTPUT_PATH}/$FOLDER/$dataset" > ${OUTPUT_PATH}/$FOLDER/$dataset/train_raw.log

        otx test --work_dir "${OUTPUT_PATH}/$FOLDER/$dataset" \
            --engine.device "gpu" > ${OUTPUT_PATH}/$FOLDER/$dataset/torch_test_raw.log

        otx export --work_dir "${OUTPUT_PATH}/$FOLDER/$dataset"

        otx test --work_dir "${OUTPUT_PATH}/$FOLDER/$dataset" \
            --checkpoint "${OUTPUT_PATH}/$FOLDER/$dataset/.latest/export/exported_model.xml" > ${OUTPUT_PATH}/$FOLDER/$dataset/ov_test_raw.log
    done
done