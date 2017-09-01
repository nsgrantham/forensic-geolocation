#!/bin/bash

FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$FILE_DIR")"

FOLDS=10
PARTS=50

DOMAIN_PATH=$PROJECT_DIR/data/external/local/domain.csv
POPULATION_PATH=$PROJECT_DIR/data/processed/local/nc-county-pop-2010-2016.csv
CENTROIDS_PATH=$PROJECT_DIR/data/external/local/nc-county-centroids.csv

KNN_N_NEIGHBORS=10
RF_N_ESTIMATORS=200
NN_EPOCHS=10
NN_BATCH_SIZE=16
DNN_EPOCHS=10
DNN_BATCH_SIZE=16
AREA_CLF_EPOCHS=10
AREA_CLF_BATCH_SIZE=16

TAXA_THRESHOLD=0.0

python src/cross-validate.py data/processed/local/ITS-outdoor.biom models/local/none-cross-val.csv \
    --seeds='none' --folds=$FOLDS --domain-fp=$DOMAIN_PATH \
    --region=0.5 --region=0.75 --region=0.9 --taxa-threshold=$TAXA_THRESHOLD \
    --weight-by='county' --weight-by-fp=$POPULATION_PATH --weight-by-col='2013' \
    --centroids='county' --centroids-fp=$CENTROIDS_PATH \
    --area='country' --area='state' --area='county' --area='city' \
    --area-clf-epochs=$AREA_CLF_EPOCHS --area-clf-batch-size=$AREA_CLF_BATCH_SIZE \
    --area-clf-verbose

python src/cross-validate.py data/processed/local/ITS-outdoor.biom models/local/coarse-cross-val.csv \
    --seeds='coarse' --folds=$FOLDS --partitions=$PARTS --domain-fp=$DOMAIN_PATH \
    --region=0.5 --region=0.75 --region=0.9 --taxa-threshold=$TAXA_THRESHOLD \
    --weight-by='county' --weight-by-fp=$POPULATION_PATH --weight-by-col='2013' \
    --area='country' --area='state' --area='county' --area='city' \
    --knn-n-neighbors=$KNN_N_NEIGHBORS --rf-n-estimators=$RF_N_ESTIMATORS \
    --nn-epochs=$NN_EPOCHS --nn-batch-size=$NN_BATCH_SIZE --nn-verbose \
    --dnn-epochs=$DNN_EPOCHS --dnn-batch-size=$DNN_BATCH_SIZE --dnn-verbose

python src/cross-validate.py data/processed/local/ITS-outdoor.biom models/local/fine-cross-val.csv \
    --seeds='fine' --folds=$FOLDS --partitions=$PARTS --domain-fp=$DOMAIN_PATH \
    --region=0.5 --region=0.75 --region=0.9 --taxa-threshold=$TAXA_THRESHOLD \
    --weight-by='county' --weight-by-fp=$POPULATION_PATH --weight-by-col='2013' \
    --area='country' --area='state' --area='county' --area='city' \
    --knn-n-neighbors=$KNN_N_NEIGHBORS --rf-n-estimators=$RF_N_ESTIMATORS \
    --nn-epochs=$NN_EPOCHS --nn-batch-size=$NN_BATCH_SIZE --nn-verbose \
    --dnn-epochs=$DNN_EPOCHS --dnn-batch-size=$DNN_BATCH_SIZE --dnn-verbose

python src/cross-validate.py data/processed/local/ITS-outdoor.biom models/local/mixed-cross-val.csv \
    --seeds='mixed' --folds=$FOLDS --partitions=$PARTS --domain-fp=$DOMAIN_PATH \
    --region=0.5 --region=0.75 --region=0.9 --taxa-threshold=$TAXA_THRESHOLD \
    --weight-by='county' --weight-by-fp=$POPULATION_PATH --weight-by-col='2013' \
    --area='country' --area='state' --area='county' --area='city' \
    --knn-n-neighbors=$KNN_N_NEIGHBORS --rf-n-estimators=$RF_N_ESTIMATORS \
    --nn-epochs=$NN_EPOCHS --nn-batch-size=$NN_BATCH_SIZE --nn-verbose \
    --dnn-epochs=$DNN_EPOCHS --dnn-batch-size=$DNN_BATCH_SIZE --dnn-verbose
