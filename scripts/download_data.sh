#!/bin/bash
mkdir -p data
cd data
kaggle competitions download -c classificacao-de-melanoma
unzip classificacao-de-melanoma.zip
rm classificacao-de-melanoma.zip
