"""
This script runs the preprocessing pipeline in sciserver-compute
1. Upload the code repo to sciserver-files.
2. Install Requirements
3. Mount Data Volumes
4. Randomize the csv based on dataSize
5. Operate on the data and apply swarp to the dataset
6. Save the result in sciserver-files.
"""
import logging
import tempfile
import os

import click

from SciServer.Authentication import login
import SciServer.Files as sf
import SciServer.Jobs as sj

from sdss.casjobs_auto import CasjobsDownloader
from utils import zip_dir

# Set logging
logger = logging.getLogger('SciServerJobSubmitter')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


class SciServerJobRunner:
    JOBS_DOMAIN = ''
    DOCKER_IMAGE = ''

    @staticmethod
    def set_job_config(domain, image_name):
        SciServerJobRunner.JOBS_DOMAIN = domain
        SciServerJobRunner.DOCKER_IMAGE = image_name
        logger.info('Set {} jobs_domain and it runs in  {} docker image'.format(domain, image_name))

    @staticmethod
    def login_sciserver(uname, passwd):
        login(uname, passwd)
        logger.info('Successfully logged in to science server')

    @staticmethod
    def download_casjobs_query(**kwargs):
        cjd = CasjobsDownloader(os.environ['SCISERVER_USERNAME'], os.environ['SCISERVER_PASSWORD'], _login=False)
        cjd.download_query(**kwargs)

    @staticmethod
    def upload_repo(file_service_name, volume_name, path_='Redshift-Resnet.tar'):
        """Archive this repo and upload it to sciserver-files"""
        tmp_dir = tempfile.TemporaryDirectory().name
        archive = zip_dir('',
                          os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'),
                          format='tar',
                          dest_dir=tmp_dir)
        logger.info('The repo is archived in {}'.format(archive))

        file_service_name = file_service_name
        file_service = sf.getFileServiceFromName(file_service_name)
        user_volumes = sf.getUserVolumesInfo(file_service)
        volume_path = user_volumes[0].get('path', 'Storage/username/persistent').rsplit('/', 1)[0] + '/' + volume_name
        logger.info('Creating {} if it doesn\'t exist.'.format(volume_path))

        sf.createUserVolume(file_service, volume_path, quiet=True)
        sf.upload(file_service,
                  path=volume_path + '/' + path_,
                  localFilePath=archive,
                  quiet=True)
        logger.info('Uploaded archived repo to sciserver-files at {}'.format(volume_path + '/' + path_))

        return volume_path + '/' + path_

    @staticmethod
    def run_preprocessing_pipeline(**kwargs):
        # TODO: change from kwargs
        """Run the preprocessing pipeline in science-server"""

        results_path = kwargs.get('results_path')
        code_tar_location = kwargs.get('code_tar_location')

        compute_domain = sj.getDockerComputeDomainFromName(SciServerJobRunner.JOBS_DOMAIN)

        jobid = sj.submitShellCommandJob('cd ../.. && mkdir -p code && rm -rf code/* '
                                         '&& tar -xvf /home/idies/workspace/{} -C code '
                                         '&& cd code/preprocessing && chmod +x preprocess.sh '
                                         '&& ./preprocess.sh'.format(code_tar_location),
                                         compute_domain,
                                         SciServerJobRunner.DOCKER_IMAGE,
                                         userVolumes=[{'name': 'AstroResearch', 'needsWriteAccess': True},
                                                      {'name': 'scratch', 'rootVolumeName': 'Temporary',
                                                       'needsWriteAccess': True}],
                                         resultsFolderPath=results_path)
        if kwargs.get('wait_for_completion'):
            sj.waitForJob(jobid, verbose=True)


@click.command('Submit preprocessing job')
@click.option('-u', '--sciserver-username',
              default=os.environ['SCISERVER_USERNAME'],
              type=str,
              help='The username for sciserver',
              show_default=True)
@click.option('-p', '--sciserver-password',
              default=os.environ['SCISERVER_PASSWORD'],
              type=str,
              help='User password for sciserver',
              show_default=True)
@click.option('-J', '--jobs-domain',
              default='Small Jobs Domain',
              type=click.Choice(['Small Jobs Domain', 'Large Jobs Domain'], case_sensitive=True),
              help='Jobs domain for the preprocessing job',
              show_default=True)
@click.option('-V', '--user-volume',
              default='AstroResearch',
              type=str,
              help='The sciserver-files volume to upload the code to (Should be in Storage volume pool)'
                   'The volume will be created if it does not exist',
              show_default=True)
@click.option('-w', '--wait-for-completion',
              is_flag=True,
              help='If true, wait for preprocessing job to complete')
def main(**kwargs):
    uname = kwargs.get('sciserver_username')
    passwd = kwargs.get('sciserver_password')
    jobs_domain = kwargs.get('jobs_domain')
    docker_image = 'SciServer Essentials'
    user_volume = kwargs.get('user_volume')
    wait_for_completion = kwargs.get('wait_for_completion')
    SciServerJobRunner.login_sciserver(uname, passwd)
    SciServerJobRunner.set_job_config(jobs_domain, docker_image)
    code_tar_location = SciServerJobRunner.upload_repo('FileServiceJHU',
                                                       volume_name=user_volume)
    SciServerJobRunner.run_preprocessing_pipeline(code_tar_location=code_tar_location,
                                                  results_path=f'/home/idies/workspace/Storage/{uname}/{user_volume}',
                                                  wait_for_completion=wait_for_completion)


if __name__ == '__main__':
    main()
