#!/bin/bash

DATASET="ZooScan_20"

# Architecture used
ARCH="resnet50"

# Method used to extract features
PRETRAIN="dinov1"

# Path to train features extracted
TRAIN_FEATURE_PATH="/features/${DATASET}/train/features_${ARCH}_${DATASET}_${PRETRAIN}.npy"
TRAIN_LABELS_PATH="/labels/${DATASET}/train/labels_${ARCH}_${DATASET}_${PRETRAIN}.npy"

# Path to test features extracted
TEST_FEATURE_PATH="/features/${DATASET}/test/features_${ARCH}_${DATASET}_${PRETRAIN}.npy"
TEST_LABELS_PATH="/labels/${DATASET}/test/labels_${ARCH}_${DATASET}_${PRETRAIN}.npy"

# Select the classifier
CLASSIFIER_TYPE="MLP" # Available choises: LogisticRegression, RandomForest, SVM, MLP
TEST_SIZE=0.2

echo "DATASET: ${DATASET}, ARCHITECTURE: ${ARCH}, PRE-TRAIN METHOD: ${PRETRAIN}, CLASSIFIER: ${CLASSIFIER_TYPE}"

######################################### Training the classifier ##########################################

python -u train_classifier.py --features_path ${TRAIN_FEATURE_PATH} --labels_path ${TRAIN_LABELS_PATH} \
                                 --classifier_type ${CLASSIFIER_TYPE} --test_size ${TEST_SIZE} --dataset_name ${DATASET}\
                                 --test_features_path ${TEST_FEATURE_PATH} --test_labels_path ${TEST_LABELS_PATH}