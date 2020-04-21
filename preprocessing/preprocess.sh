#!/usr/bin/env bash
conda update -yq conda
conda env create --file sciserver-environment.yml
git clone https://github.com/sciserver/SciScript-Python.git \
	&& cd SciScript-Python \
	&& git checkout sciserver-v2.0.13 \
	&& cd py3 \
	&& (source activate preprocess-env && python setup.py install)

source activate preprocess-env
cd ..
cd ..
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
python3 sdss/preprocess.py data/SDSS_DR12.csv -u utimalsina -C 500 -N 100000 -O 64 --checkpoint-dir ckpts --num-processes=32 --volume-name=AstroResearch
cd ..
