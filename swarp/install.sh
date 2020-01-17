#!/usr/bin/env bash

#ToDo: Add Options to specify directories from command line
#ToDo: Add UsageScript.
INSTALL_DIR = $HOME

if [[ $# > 0 ]];
    then
        INSTALL_DIR=$1
fi

SOURCEFILE=${INSTALL_DIR}/.bashrc   # Where to add swarp to path

# REMOVE DIRECTORY IF EXISTS

# SCRATCH
SCRATCH_DIR=${INSTALL_DIR}/.swarp/scratch

# SWARPDIR
SWARP_DIR=${INSTALL_DIR}/.swarp

rm -rf ${SWARP_DIR} 2> /dev/null

# MAKE DIRECTORY
mkdir -p ${SCRATCH_DIR}

# DOWNLOAD URL
DOWNLOAD_URL="https://www.astromatic.net/download/swarp/swarp-2.38.0.tar.gz"

# DOWNLAOD and EXTRACT
wget ${DOWNLOAD_URL} -O ${SCRATCH_DIR}/swarp-2.38.0.tar.gz

tar -xf ${SCRATCH_DIR}/swarp-2.38.0.tar.gz -C $SCRATCH_DIR

# Go to the scratch directory
cd ${SCRATCH_DIR}/swarp-2.38.0

./configure --prefix=${SWARP_DIR};

if [[ $? != 0 ]];
 then
    echo "Configuration failed for this machine";
    exit 1;
fi

echo INSTALLING SWARP;
make install;
echo SWARP INSTALLATION COMPLETE;

rm -rf ${SCRATCH_DIR};
#
if [[ -d ${SWARP_DIR}/bin && !( $PATH =~ .*${SWARP_DIR}/bin.*) ]];
    then
        echo  "Adding SWARP Binary To PATH"
        echo  "export PATH=\$PATH:${SWARP_DIR}/bin" >> ${SOURCEFILE}
        source ${SOURCEFILE}
        echo Installation Complete, swarp has been added to the path
        exit 0;
fi

echo Installation Complete, SWARP path already exists
exit 0;


