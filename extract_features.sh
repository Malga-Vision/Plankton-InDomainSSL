#!/bin/bash

# Dataset folder name to use for pre-training
DATASET="ZooScan_20"

# Path to the train dataset folder
DIR="/datasets/${DATASET}/train"

# Path to the test dataset folder
TEST_DIR="/datasets/${DATASET}/test"

# Architecture to use
ARCH="resnet50"
WORKERS=2

# Pre-trained method to use
PRETRAINED="dinov1"


############################################## Pre-trained model paths ####################################################

# DeepCluster v1 pretrain on ZooScan93 ResNet50
# PRETRAINED_MODEL_PATH="/pretrained_models_ckp/checkpoint_resnet50.pth.tar"

# DeepCluster v1 pretrain on WHOI80 ResNet50
# PRETRAINED_MODEL_PATH="/pretrained_models_ckp/deepcluster_resnet50_WHOI80.pth.tar"

# DeepCluster v1 pretrain on Imagenet ResNet50
# PRETRAINED_MODEL_PATH="/pretrained_models_ckp/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k-bb8681e2.pth"

# DeepCluster v2 pretrain on Imagenet ResNet50
PRETRAINED_MODEL_PATH="/pretrained_models_ckp/deepclusterv2_400ep_2x224_pretrain.pth.tar"

# DeepCluster v1 pretrain on WHOI15 ResNet50
# PRETRAINED_MODEL_PATH="/pretrained_models_ckp/checkpoint_resnet50.pth.tar"

# DeepCluster v1 pretrain on WHOI22 ResNet50
# PRETRAINED_MODEL_PATH="/pretrained_models_ckp/checkpoint_resnet50.pth.tar"



# Text file to save outputs
OUTPUT_FILE="output.txt"

# Loop to execute the code 3 times to get the average accuracy

for i in 1 2 3 ; do
        
    ################################# Imagenet Supervised #################################


    if [ "$PRETRAINED" == "imagenet" ]; then

        python feature_extractor.py --data ${DIR} --imagenet --arch ${ARCH} --test_data ${TEST_DIR} \
                        --workers ${WORKERS} --dataset_name ${DATASET} --pretrain_method ${PRETRAINED} 

    fi

    ################################# DeepCluster #################################

    if [ "$PRETRAINED" == "deepcluster" ]; then

        python feature_extractor.py --data ${DIR} --deepcluster --model ${PRETRAINED_MODEL_PATH} --arch ${ARCH} --test_data ${TEST_DIR}\
                            --workers ${WORKERS} --dataset_name ${DATASET} --pretrain_method ${PRETRAINED} #--verbose #--head ${HEAD}
    fi

    ################################# Dino v1 #################################

    if [ "$PRETRAINED" == "dinov1" ]; then

        # Path to the ResNet50 pre-trained with Dino v1
        PRETRAINED_MODEL_PATH="/pretrained_models_ckp/dino_resnet50_200ep_ZOOSCAN93.pth"

        python feature_extractor.py --data ${DIR} --model ${PRETRAINED_MODEL_PATH} --arch ${ARCH} --test_data ${TEST_DIR}\
                            --workers ${WORKERS} --dataset_name ${DATASET} --pretrain_method ${PRETRAINED} \
    fi

    ################################# Moco v1 #################################

    if [ "$PRETRAINED" == "mocov1" ]; then

        # Path to the ResNet50 pre-trained with MoCo v1
        PRETRAINED_MODEL_PATH="/pretrained_models_ckp/moco_v1_200ep_pretrain.pth.tar"

        python feature_extractor.py --data ${DIR} --model ${PRETRAINED_MODEL_PATH} --arch ${ARCH} --test_data ${TEST_DIR}\
                            --workers ${WORKERS} --dataset_name ${DATASET} --pretrain_method ${PRETRAINED}
    fi

    ################################# Moco v2 #################################

    if [ "$PRETRAINED" == "mocov2" ]; then

        # Path to the ResNet50 pre-trained with MoCo v2
        PRETRAINED_MODEL_PATH="/pretrained_models_ckp/moco_v2_800ep_pretrain.pth.tar"

        python feature_extractor.py --data ${DIR} --model ${PRETRAINED_MODEL_PATH} --arch ${ARCH} --test_data ${TEST_DIR}\
                            --workers ${WORKERS} --dataset_name ${DATASET} --pretrain_method ${PRETRAINED}
    fi

    ./train_classifier.sh | tee -a ${OUTPUT_FILE}

done