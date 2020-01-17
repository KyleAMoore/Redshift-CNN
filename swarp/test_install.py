import os

from SciServer.Authentication import login
from SciServer.Files import getFileServiceFromName, getFileServicesNames, createUserVolume, upload
from SciServer.Jobs import getDockerComputeDomains, submitShellCommandJob


def _login(uname=None, passwd=None):
    """login to sciserver"""
    token = login(uname, passwd)
    return token


def upload_file(volume, filepath, src_file=None):
    """Upload any file to the Sciserver-Files Service"""
    available_services = getFileServicesNames()
    file_service_name = available_services[0]['name']
    _path = os.path.join(volume, filepath)
    upload(file_service_name, path=_path, localFilePath=src_file)
    return _path


def create_user_volume(volume_name='AstroResearch'):
    """Create a separate user volume for testing this installation"""
    available_services = getFileServicesNames()
    file_service_name = available_services[0]['name']
    file_service = getFileServiceFromName(file_service_name)
    user_volume_path = '/'.join([file_service['rootVolumes'][0]['name'],
                                             os.environ['SCISERVER_USERNAME'],
                                             volume_name])
    createUserVolume(file_service, user_volume_path)
    return user_volume_path


if __name__ == '__main__':
    from pprint import pprint
    _login(uname=os.environ['SCISERVER_USERNAME'], passwd=os.environ['SCISERVER_PASSWORD'])
    created_volume = create_user_volume()
    script_path = upload_file(created_volume, 'scripts/install.sh', src_file='install.sh')