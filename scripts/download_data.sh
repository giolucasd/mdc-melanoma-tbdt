#!/bin/bash
mkdir -p data
cd data
curl -L -o ./isic-2019-jpg-224x224-resized.zip https://www.kaggle.com/api/v1/datasets/download/nischaydnk/isic-2019-jpg-224x224-resized
unzip isic-2019-jpg-224x224-resized.zip
rm isic-2019-jpg-224x224-resized.zip
