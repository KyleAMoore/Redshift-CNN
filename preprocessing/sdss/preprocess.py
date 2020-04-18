import os
from pathlib import Path

import math

from tempfile import mkdtemp

from datetime import datetime

import shutil

import subprocess

from collections import deque

from multiprocessing import Pool

from SciServer.Config import isSciServerComputeEnvironment

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.lupton_rgb import make_lupton_rgb

import numpy as np
import pandas as pd

import click

from logger_factory import LoggerFactory
from constants import SAS_URL, SWARP_COMMAND
from sdss_utils import get_guid
from checkpoint_objects import CheckPoint, RedShiftCheckPointObject


class PreProcess:
    def __init__(self,
                 images_meta,
                 seed=32,
                 uname='',
                 config_file='.swarp.conf',
                 num_samples=100,
                 output_image_size=64,
                 checkpoint_steps=10,
                 num_processes=10,
                 checkpoint_dir=os.getcwd(),
                 overwrite_checkpoints=True):
        """ A generic pre-process class to convert sdss images fits to a datacube with redshift values
        Note: This class assumes that a swarp installation is available in your system.
        :param config_file: config file for swarp
        :param images_meta: the meta data frame/csv file
        :param seed: the random seed for numpy
        :param uname: username for sciserver
        :param num_samples: number of galaxies to perform pre-processing on
        :param output_image_size: the size of the output image
        :param checkpoint_steps: checkpoint after completing pre-processing for this amount of galaxies
        :param num_processes: the number of processes to use in the swarp wrapper pool
        :param checkpoint_dir: the location of the checkpoints
        :param overwrite_checkpoints: If True, overwrite the exsisting checkpoints
        """
        np.random.seed(seed)
        self.logger = LoggerFactory.get_logger(self.__class__.__name__,
                                               'DEBUG',
                                               add_file_handler=True,
                                               filename='preprocess.log')
        self.url = SAS_URL
        self.images_meta = images_meta
        if isinstance(images_meta, str) and images_meta.endswith('.csv'):
            self.images_meta = pd.read_csv(images_meta)
        self.galaxies = self._randomize_meta(num_samples)
        self.output_image_size = output_image_size
        self.checkpoint_steps = checkpoint_steps
        self.swarp_config_file = Path(__file__).parent.resolve() / config_file
        self.uname = uname
        self.checkpoint_dir = checkpoint_dir
        self.overwrite_checkpoints = overwrite_checkpoints
        self.num_processes = num_processes

        if isSciServerComputeEnvironment():
            if uname is None:
                raise Exception('Please provide a username for science server.')
            self.fits_download_loc = '/home/idies/workspace/Temporary/{}/scratch'.format(self.uname)
            self.checkpoint_dir = '/home/idies/workspace/Temporary/{}/scratch/{}'.format(self.uname, checkpoint_dir)

        else:
            self.fits_download_loc = mkdtemp()

        self.checkpoint_blocks = self._form_checkpoint_blocks()

    def _randomize_meta(self, num_samples):
        """Randomly select num_samples entry from the metadata"""
        self.logger.debug('Sampling the data frame to {} galaxies'.format(num_samples))
        return self.images_meta.sample(frac=num_samples / self.images_meta.shape[0])

    def run(self):
        rundir = Path(self.checkpoint_dir)
        if not rundir.resolve().exists():
            os.makedirs(rundir)
        if not self.overwrite_checkpoints:
            shutil.rmtree(rundir / 'galaxies', ignore_errors=True)
        os.makedirs(rundir / 'galaxies', exist_ok=True)
        self.logger.debug('Created a directory called {} to save lupton-rgb images'
                          .format(rundir / 'galaxies'))
        process_pool = PreProcess.get_process_pool(self.num_processes)

        [process_pool.apply_async(self._run_preprocess_for_one_ckpt,
                                  kwds={'ckpt_info': x},
                                  callback=self._save_ckpt,
                                  error_callback=self._on_ckpt_fail) for x in self.checkpoint_blocks]
        process_pool.close()
        process_pool.join()

    def _on_ckpt_fail(self, exception):
        self.logger.error('Error. {}'.format(exception))

    def _save_ckpt(self, obj_guid):
        if obj_guid == 'SKIP_SENTINEL':
            return
        self.logger.info('Completed preprocessing for galaxies (GUID): {}'.format(obj_guid['guid']))
        ckpt = CheckPoint(self.checkpoint_dir, obj_guid['obj'], obj_guid['guid'])
        ckpt.save_checkpoint(overwrite=True)
        self.logger.info('Checkpoint saved as {}'.format(ckpt.get_loc()))

    def _run_preprocess_for_one_ckpt(self, ckpt_info):
        galaxies = self.galaxies[ckpt_info['start_idx']: ckpt_info['end_idx']]
        guid = ckpt_info['guid']
        return self._run_preprocess(galaxies, guid)

    def _form_checkpoint_blocks(self):
        rows = self.galaxies.shape[0]
        num_checkpoints = math.ceil(rows / self.checkpoint_steps)
        if num_checkpoints == 0:
            num_checkpoints = 1
        self.logger.debug('{} Galaxies with {} checkpoint steps form will form {} checkpoints'
                          .format(rows, self.checkpoint_steps, num_checkpoints))
        checkpoint_blocks = deque()
        for i in range(num_checkpoints):
            start_idx = i * self.checkpoint_steps
            end_idx = (i + 1) * self.checkpoint_steps
            if end_idx >= self.galaxies.shape[0]:
                end_idx = self.galaxies.shape[0]
            checkpoint_block = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'guid': get_guid(self.galaxies[start_idx: end_idx], col_name='specObjID')
            }
            checkpoint_blocks.append(checkpoint_block)
        self.logger.debug('Successfully initialized checkpoint blocks')
        return checkpoint_blocks

    def _run_preprocess(self, galaxies, guid):
        dl_count = 0
        galaxy_count = 0
        run_dir = os.getcwd()
        if CheckPoint.checkpoint_exists(self.checkpoint_dir, guid) and not self.overwrite_checkpoints:
            self.logger.debug('Checkpoint with GUID {} already exists. Skipping'.format(guid))
            return 'SKIP_SENTINEL'
        else:
            CheckPoint.remove_ckpt(self.checkpoint_dir, guid)
        redshift_objects = []
        for i, galaxy in galaxies.iterrows():
            download_urls = self._get_formatted_urls(galaxy['rerun'],
                                                     galaxy['run'],
                                                     galaxy['camcol'],
                                                     galaxy['field'])
            files = []
            # Download the images
            for url in download_urls:
                filename = url.split('/')[-1]
                files.append(self.fits_download_loc + '/' + filename)

                if Path(self.fits_download_loc).joinpath(filename.replace('.bz2', '')).exists():
                    self.logger.debug('Compressed file exists, skipping download.')
                else:
                    self.logger.debug('Downloading compressed file {}'
                                      ' from the sdss url to {}'.format(filename,
                                                                        self.fits_download_loc + '/' + filename))
                    subprocess.run('wget {0} -O {1}'.format(url,
                                                            self.fits_download_loc + '/' + filename),
                                   shell=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   check=True)
                dl_count += 1

            os.chdir(self.fits_download_loc)
            # Check all the files exist
            assert [os.path.exists(file) or os.path.exists(file.replace('.bz2', ''))
                    for file in files] == [True] * 5, 'Compressed or uncompressed files  missing'

            # Extract the fits files
            fits_files = []
            for file in files:
                if Path(file).exists():
                    op = subprocess.run('bzip2 -dkf {}'.format(file),
                                        capture_output=False,
                                        shell=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        check=False)
                    assert op.returncode == 0, f'Error decompressing the fits file {file}'
                    os.remove(file)
                else:
                    assert Path(file.replace('.bz2', '')).exists(), f"Uncompressed file doesn't exist"
                fits_files.append(file.replace('.bz2', ''))

            # Check if the files exist
            assert [os.path.exists(file) for file in fits_files] == [True] * 5, 'File {} doesnot exist'.format(file)
            assert [file.endswith('.fits') for file in fits_files]
            self.logger.debug('Successfully uncompressed files for this galaxy')

            data_mat = self._apply_swarp(galaxy, fits_files, cleanup=True)
            os.chdir(run_dir)
            image_filename = '{0}/{1}-{2}-{3}.jpg'.format(os.path.join(self.checkpoint_dir, 'galaxies'),
                                                          'galaxy', galaxy['specObjID'], galaxy['z'])
            make_lupton_rgb(data_mat[:, :, 3],
                            data_mat[:, :, 2],
                            data_mat[:, :, 1],
                            Q=8, stretch=0.4,
                            filename=image_filename)
            self.logger.debug('Galaxy image saved as {}'.format(image_filename))
            assert dl_count == 5, 'Downloaded only {} fits files'.format(dl_count)

            self.logger.debug('Completed pre-processing for this galaxy with redshift value {} at index {}'
                              .format(galaxy['z'], galaxy_count))
            redshift_objects.append(RedShiftCheckPointObject(np_array=data_mat,
                                                             redshift=galaxy['z'],
                                                             galaxy_meta=galaxy,
                                                             image=image_filename,
                                                             timestamp=datetime.now()))
            galaxy_count += 1
            dl_count = 0
        return {'obj': redshift_objects, 'guid': guid}

    def _apply_swarp(self, galaxy, fits_files, cleanup=True):
        """Apply the swarp tool for this galaxy and return a preprocessed matrix"""
        center = self._hmsdms_string(galaxy['ra'], galaxy['dec'])
        data_mat = None
        self.logger.debug('The galaxy is centered at {0},{1}'.format(*center))
        for i, fits_file in enumerate(fits_files):
            ret = PreProcess.run_swarp_subprocess(
                fits_file=fits_file,
                config_file=self.swarp_config_file,
                center=center,
                image_out='coadd-preprocess{}-{}.fits'.format(os.getpid(), i + 1),
                weight_out='coadd.weight-preporcess{}-{}.fits'.format(os.getpid(), i + 1))
            if ret != 0:
                raise Exception('Error processing the input image')

            with fits.open('coadd-preprocess{}-{}.fits'.format(os.getpid(), i + 1)) as _fits_data:
                one_channel = np.expand_dims(_fits_data[0].data, axis=-1)
                if data_mat is None:
                    data_mat = one_channel
                else:
                    data_mat = np.concatenate((data_mat, one_channel), axis=-1)
                if cleanup:
                    os.remove('coadd-preprocess{}-{}.fits'.format(os.getpid(), i + 1))
        return data_mat

    def _get_formatted_urls(self, rerun, run, camcol, field):
        """Get formatted urls for this galaxy"""
        possible_bands = ['u', 'g', 'r', 'i', 'z']
        run_arr = list(str(run).strip())
        if len(run_arr) < 6:
            run_arr = ['0'] * (6 - len(run_arr)) + run_arr
        run_str = ''.join(run_arr)

        field_arr = list(str(field).strip())
        if len(field_arr) < 4:
            field_arr = ['0'] * (4 - len(field_arr)) + field_arr

        field_str = ''.join(field_arr)
        urls = []
        for band in possible_bands:
            urls.append(self.url.format(run=str(run),
                                        run_str=run_str,
                                        rerun=str(rerun),
                                        field=field_str,
                                        camcol=str(camcol),
                                        band=band))
        return urls

    def _hmsdms_string(self, ra, dec):
        """Retrun hmsdms string from ra, dec"""
        center_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        coords = center_coord.to_string('hmsdms').split(' ')
        for i in range(len(coords)):
            coords[i] = coords[i].replace('h', ':')
            coords[i] = coords[i].replace('m', ':')
            coords[i] = coords[i].replace('s', '')
            coords[i] = coords[i].replace('d', ':')
        return tuple(coords)

    @staticmethod
    def run_swarp_subprocess(fits_file,
                             config_file,
                             center,
                             image_out,
                             weight_out):
        try:
            subprocess.run(
                args='swarp '
                     f'{fits_file}[0] '
                     f' -c {config_file} '
                     f' -CENTER {center[0]},{center[1]} '
                     f' -IMAGEOUT_NAME {image_out} '
                     f' -WEIGHTOUT_NAME {weight_out} ',
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                capture_output=False,
                check=True)
        except subprocess.CalledProcessError:
            return -1
        return 0

    @staticmethod
    def get_process_pool(num):
        return Pool(processes=num)


@click.command(name='Run Preprocessing Pipeline')
@click.argument('images_meta')
@click.option('-c', '--config-file',
              default='.swarp.conf',
              type=str,
              help="The default swarp-config file to use")
@click.option('-s', '--seed',
              default=42,
              type=int,
              help="The random seed for numpy")
@click.option('-u', '--uname',
              default='',
              type=str,
              help="The science-server username, Please provide this for running is sci-server")
@click.option('-C', '--checkpoint-steps',
              default=100,
              type=int,
              help="The number of galaxies to include in a single checkpoint")
@click.option('-N', '--num-samples',
              default=1000,
              type=int,
              help='The number of galaxies to run pre-processing on.')
@click.option('-O', '--output-image-size',
              default=64,
              type=int,
              help='The output image size')
@click.option('-o', '--overwrite-checkpoints',
              is_flag=True,
              help='If set, overwrite the existing checkpoints')
@click.option('--checkpoint-dir',
              default=os.getcwd(),
              type=str,
              help='The checkpoint directory')
@click.option('--num-processes',
              default=10,
              type=int,
              help='The number of processes to use (uses multiprocessing.Pool)')
def main(**kwargs):
    cs = PreProcess(**kwargs)
    cs.run()


if __name__ == '__main__':
    main()
