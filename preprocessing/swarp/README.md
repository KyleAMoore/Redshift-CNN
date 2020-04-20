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
./install.sh $HOME $HOME/.bashrc
```
If ths script runs without any issue, then swarp has been install and will now be 
available as a user command.

## Installing SWARP in sciserver-compute
The file [`test_install.py`](test_install.py) tries to install swarp in 
[`sciserver-compute`](https://apps.sciserver.org/compute/jobs) environment. 
This will be helpful to leverage the massive computing platform that is provided by `sciserver`.
 
To run this test please install [`SciScript-Python`](https://github.com/sciserver/SciScript-Python) according to the 
installation instructions in the repository. *Note: the github install instructions don't install 
requirements for SciScript-Python, `requests` package needs to be installed manually.* 

In order to run the test you also need to provide the following as your environment variables:

1. SCISERVER_USERNAME: `username for sciserver`
2. SCISERVER_PASSWORD: `password for sciserver`

Now, that everything has been setup you can run the test by:
```bash
python test_install.py
```

By default, `test_install.py` will create a `UserVolume` called `AstroResearch` to install swarp.
The script [install.sh](install.sh) will be uploaded to `AstroResearch/scripts`. Swarp binary 
could be found in `AstroResearch/.swarp`. You can clean this up from the sciserver-dashboard. 

After the script has finished installing swarp, you should get a following message:
```bash
Waiting...
Waiting...
Waiting...
Waiting...
Waiting...
Waiting...
Waiting...
Waiting...
Done!
{'status': 32, 'statusMeaning': 'SUCCESS', 'jobId': ****}
```
If the `statusMeaning` is `SUCCESS`, then, swarp has been setup and now can be used in any of your further compute pipelines. 





