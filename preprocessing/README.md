# Preprocessing SDSS Galaxy Images 
This directory contains code to perform a CASJOBS Query and download SDSS metadata as a CSV File for galaxies in the
SDSS survey, Download the fits files for each SDSS band `u`, `g`, `r`, `i`, `z` for the galaxies returned from the queries
and preprocess them into a collection of datacube of pixels centered around a particular galaxy along with its redshift value.

## Contents
1. [__sdss__](./sdss) : Contains the code for downloading images and applying resampling using the swarp tool in a local 
machine or `sciserver-compute` environment
2. [__swarp__](./swarp) : Contains utilities and file helpers for aiding with the swarp installation
3. [__tests__](./tests) : Contains tests

## Before Running
Use the [casjobs](http://skyserver.sdss.org/CasJobs) server to run the following query (We used the DR12 context) and 
download the CSV file and place it in the [data](./data) directory. 
```sql
SELECT za.specObjID, za.bestObjID, za.class, za.subClass, za.z, za.zErr,
  po.objID, po.type, po.flags, po.ra, po.dec,
  (po.petroMag_r-po.extinction_r) as dered_petro_r,
  zp.z as zphot, zp.zErr as dzphot,
  zi.e_bv_sfd,zi.primtarget, zi.sectarget,zi.targettype,zi.spectrotype,zi.subclass

INTO MyDB.SDSS_DR12
FROM SpecObjAll za
  JOIN PhotoObjAll po ON (po.objID = za.bestObjID)
  JOIN Photoz zp ON (zp.objID = za.bestObjID)
  JOIN galSpecInfo zi ON (zi.SpecObjID = za.specObjID)
WHERE
  (za.z>0 AND za.zWarning=0)
    AND (za.targetType='SCIENCE' AND za.survey='sdss')
    AND (za.class='GALAXY' AND zi.primtarget>=64)
    AND (po.clean=1 AND po.insideMask=0)
  AND ((po.petroMag_r-po.extinction_r)<=17.8)
  AND za.z <= 0.4
```

## How to run this code?

1. In __Sciserver-Compute__

    The file [submit_preprocessing_job.py](submit_preprocessing_job.py) archives this directory as a tar file
and uploads it in a `UserVolume` in SciServer-Files. After that, it uses a shell command job from 
`sciserver-compute`, to extract the archive, and submit [`preprocess.sh`](preprocess.sh) as a shell command job.
__preprocess.sh__ first creates a conda environment to install all the dependencies for the project in `sciserver-compute`
and runs the preprocessing code at [preprocess.py](sdss/preprocess.py). The preprocess.py can be called with the following 
options:
    ```bash
    Usage: preprocess.py [OPTIONS] IMAGES_META
    
    Options:
      -c, --config-file TEXT          The default swarp-config file to use
      -s, --seed INTEGER              The random seed for numpy
      -u, --uname TEXT                The science-server username, Please provide
                                      this for running is sci-server
    
      -C, --checkpoint-steps INTEGER  The number of galaxies to include in a
                                      single checkpoint
    
      -N, --num-samples INTEGER       The number of galaxies to run pre-processing
                                      on.
    
      -O, --output-image-size INTEGER
                                      The output image size
      -o, --overwrite-checkpoints     If set, overwrite the existing checkpoints
      --checkpoint-dir TEXT           The checkpoint directory
      --volume-name TEXT              The sciserver-volume in which checkpoint
                                      directory should reside(should be in Storage
                                      pool)
    
      --num-processes INTEGER         The number of processes to use (uses
                                      multiprocessing.Pool)
    
      --help                          Show this message and exit.
    ```
    __Note__: To customize the arguments passed to preprocess.py in `sciserver-compute`, change [`preprocess.sh`](./preprocess.sh).

2.  __Locally__

    To run this code locally, use the preprocess.py with specific options(as shown above) to 
from this directory. Fits images are downloaded in a temporary directory provided by your operating system.