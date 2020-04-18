import os
import logging

from SciServer.Authentication import login
from SciServer.Files import getFileServiceFromName, getFileServicesNames, createUserVolume, upload
from SciServer.Jobs import (submitShellCommandJob,
                            getDockerComputeDomainFromName,
                            waitForJob,
                            getJobStatus)

DIR_PREFIX = '/home/idies/workspace/'


def _login(uname=None, passwd=None):
    """login to sciserver"""
    token = login(uname, passwd)
    return token


def upload_swarp_script(volume, filepath, src_file=None):
    """Upload any file to the Sciserver-Files Service"""
    available_services = getFileServicesNames()
    file_service_name = available_services[0]['name']
    _path = os.path.join(volume, filepath)
    upload(file_service_name, path=_path, localFilePath=src_file)
    return _path


def install_swarp(install_script_path, volume):
    """Submit the installation script and wait for it

    Parameters
    ----------
    install_script_path: path in the actual sciserver-filesystem that points to the script
    volume: This is the volume where the swarp will be installed. Note the DIR_PREFIX
    """
    # ToDo: Build upon this function to incorporate more options like changing compute and other options.
    large_jobs_domain = getDockerComputeDomainFromName('Small Jobs Domain')
    install_dir = DIR_PREFIX + volume
    job_id = submitShellCommandJob('bash {0} {1} nofile.txt'.format(DIR_PREFIX + install_script_path, install_dir),
                                   dockerComputeDomain=large_jobs_domain,
                                   dockerImageName='SciServer Essentials',
                                   userVolumes=[{'name': 'AstroResearch', 'needsWriteAccess': True}],
                                   dataVolumes=None,
                                   resultsFolderPath=DIR_PREFIX + volume + '/JobsLog',
                                   jobAlias='Test_Swarp_Installation')

    waitForJob(job_id, verbose=True)
    print(getJobStatus(job_id))


def create_user_volume(volume_name='AstroResearch'):
    """Create a separate user volume for testing this installation"""
    available_services = getFileServicesNames()
    file_service_name = available_services[0]['name']
    file_service = getFileServiceFromName(file_service_name)
    user_volume_path = '/'.join([file_service['rootVolumes'][0]['name'],
                                 os.environ['SCISERVER_USERNAME'],
                                 volume_name])
    createUserVolume(file_service, user_volume_path)
    print(user_volume_path)
    return user_volume_path


if __name__ == '__main__':
    _login(uname=os.environ['SCISERVER_USERNAME'], passwd=os.environ['SCISERVER_PASSWORD'])
    created_volume = create_user_volume()
    script_path = upload_swarp_script(created_volume, 'scripts/install.sh', src_file='install.sh')
    install_swarp(script_path, created_volume)
