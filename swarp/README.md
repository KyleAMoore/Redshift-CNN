# SWARP Installation
[Swarp](https://www.astromatic.net/software/swarp) is an image resampling tool that can be used to resample FITS images. In
this directory, there is a installation script that tries to install swarp as a 
user in your home directory.

## Installing swarp as a user
This directory contains a shell script to install swarp tool as a user.
You can manually change the target directory to install swarp in. By default,
this script tries to install swarp in `${HOME}/.swarp` and adds swarp to the `$PATH`
environment variable.

```bash
chmod +x install.sh
./install.sh
```
If ths script runs with any issue, then swarp has been install and will now be 
available as a user command.

## Installing SWARP in sciserver-compute
The file [`test_install.py`](./test_install.py) tries to install swarp in 
[`sciserver-compute`](https://apps.sciserver.org/compute/jobs) environment. 
This will be helpful to leverage the massive computing platform that is provided by `sciserver`.
 
To run this test please install [`SciScript-Python`](https://github.com/sciserver/SciScript-Python) according to the 
installation instructions in the repository.

In order to run the test you also need to provide the following as your environment variables:

1. SCISERVER_USERNAME: `username for sciserver`
2. SCISERVER_PASSWORD: `password for sciserver`


