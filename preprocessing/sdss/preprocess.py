import os
from pathlib import Path

import math

from tempfile import mkdtemp

from datetime import datetime

import shutil

import subprocess

from collections import deque

from multiprocessing import Pool, Manager

from SciServer.Config import isSciServerComputeEnvironment

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.lupton_rgb import make_lupton_rgb

import numpy as np
import pandas as pd

import click

from logger_factory import LoggerFactory
from constants import SAS_URL, SWARP_COMMAND, CKPT_GUID
from sdss_utils import get_guid
from checkpoint_objects import CheckPoint, RedShiftCheckPointObject


class PreProcess:
    """A generic preprocess class to convert sdss images fits to a datacube with redshift values

    Note: This class assumes that a swarp installation is available in your system.

    Parameters
    ----------
    images_meta : str or pd.DataFrame
        the meta data frame/csv file
    uname : str
        username for sciserver
    seed : str, default=32
        the random seed for numpy
    num_samples : int, default=100
        number of galaxies to perform pre-processing on
    output_image_size : int, default=64
        the size of the output image, assuming a square frame
    config_file : str, default=.swarp.conf
        config file for swarp tool
    checkpoint_steps : int, default=10
        checkpoint after completing pre-processing for this amount of galaxies
    num_processes : int, default=10
        the number of processes to use for this Propercess operation
    checkpoint_dir : str, os.path like, default=os.getcwd()
        the location of the checkpoints
    volume_name: str, default=AstroResearch
        If running in `sciserver-compute` name of the volume in
        storage volume pool to create checkpoints in
    overwrite_checkpoints: bool, default=False
        If True, overwrite the existing checkpoints
    """
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
                 volume_name='AstroResearch',
                 overwrite_checkpoints=False):
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
        self.volume_name = volume_name

        if isSciServerComputeEnvironment():
            if uname is None:
                raise Exception('Please provide a username for science server.')
            self.fits_download_loc = '/home/idies/workspace/Temporary/{}/scratch'.format(self.uname)
            self.checkpoint_dir = '/home/idies/workspace/Storage/{}/{}/{}'.format(self.uname,
                                                                                  self.volume_name,
                                                                                  checkpoint_dir)

        else:
            self.fits_download_loc = mkdtemp()

        self.process_blocks = self._form_process_blocks()
        self.counter = 0

    def _randomize_meta(self, num_samples):
        """Randomly select num_samples entry from the metadata"""
        self.logger.debug('Sampling the data frame to {} galaxies'.format(num_samples))
        return self.images_meta.sample(frac=num_samples / self.images_meta.shape[0])

    def _form_process_blocks(self):
        """This method divides galaxy metadata into blocks for each process

        Notes
        -----
            This is primarily done to offshoot heavy switching of contexts in a process pool
            so that each process can operate on distinct number of galaxies rather
            than a single galaxy
        """
        rows = self.galaxies.shape[0]
        num_blocks = math.ceil(rows / self.checkpoint_steps)
        if num_blocks == 0:
            num_blocks = 1
        self.logger.debug('{} Galaxies with {} checkpoint steps form will form {} blocks'
                          .format(rows, self.checkpoint_steps, num_blocks))
        process_blocks = deque()
        for i in range(num_blocks):
            start_idx = i * self.checkpoint_steps
            end_idx = (i + 1) * self.checkpoint_steps
            if end_idx >= self.galaxies.shape[0]:
                end_idx = self.galaxies.shape[0]
            checkpoint_block = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'guid': get_guid(self.galaxies[start_idx: end_idx], col_name='specObjID')
            }
            process_blocks.append(checkpoint_block)
        self.logger.debug('Successfully initialized process blocks')
        return process_blocks

    def run(self):
        """Run the preprocessing pipeline

        This method runs the preprocessing pipeline for given samples
        by spinning up processes provided while initializing the class
        """
        rundir = Path(self.checkpoint_dir)
        if not rundir.resolve().exists():
            os.makedirs(rundir)
        if self.overwrite_checkpoints:
            shutil.rmtree(rundir / 'galaxies', ignore_errors=True)
        os.makedirs(rundir / 'galaxies', exist_ok=True)
        self.logger.debug('Created a directory called {} to save lupton-rgb images'
                          .format(rundir / 'galaxies'))
        guid = CKPT_GUID
        if self.overwrite_checkpoints:
            CheckPoint.remove_ckpt(self.checkpoint_dir, guid)
        if not CheckPoint.checkpoint_exists(self.checkpoint_dir, guid):
            checkpoint = CheckPoint(self.checkpoint_dir, RedShiftCheckPointObject, guid)
            checkpoint.save_checkpoint()

        process_pool = PreProcess.get_process_pool(self.num_processes)
        manager = Manager()
        counter = manager.Value('i', 0)
        checkpoint_objects = manager.Queue()
        [process_pool.apply_async(self._run_preprocess_for_one_block,
                                  kwds={'ckpt_info': x,
                                        'counter': counter,
                                        'ckpt_objs': checkpoint_objects},
                                  callback=self._on_process_complete,
                                  error_callback=self._on_process_fail) for x in self.process_blocks]
        process_pool.close()
        process_pool.join()

    def _on_process_complete(self, queue_remaining):
        """Callback method for successful processes"""
        if not queue_remaining.empty():
            self.logger.debug('Saving remaining objects to the checkpoint')
            self.save_checkpoint(queue_remaining, CKPT_GUID)
        self.logger.info(f'Process (id: {os.getpid()}) completed.')

    def _on_process_fail(self, exception):
        """Callback method for failed processes"""
        self.logger.error(f'Error on process with pid {os.getpid()}, {exception}')

    def _run_preprocess_for_one_block(self, ckpt_info, counter, ckpt_objs):
        """Run the preprocess pipeline for one checkpoint step"""
        galaxies = self.galaxies[ckpt_info['start_idx']: ckpt_info['end_idx']]
        guid = CKPT_GUID
        return self._run_preprocess(galaxies, guid, counter=counter, ckpt_objs=ckpt_objs)

    def _run_preprocess(self, galaxies, guid, counter, ckpt_objs):
        """Run preprocessing pipeline for given galaxies."""
        dl_count = 0
        galaxy_count = 0
        run_dir = os.getcwd()
        redshift_objects = ckpt_objs
        objects_on_disk = CheckPoint.get_object_set(self.checkpoint_dir, guid)
        last_modified = CheckPoint.last_modified(self.checkpoint_dir, guid)
        self.logger.debug(f'Objects on disk: {len(objects_on_disk)}, last Modified: {last_modified}')
        for i, galaxy in galaxies.iterrows():
            if CheckPoint.last_modified(self.checkpoint_dir, guid) > last_modified:
                self.logger.debug('Loading new checkpoint, as the last checkpoint was updated.')
                objects_on_disk = CheckPoint.get_object_set(self.checkpoint_dir, guid)
                last_modified = CheckPoint.last_modified(self.checkpoint_dir, guid)
                self.logger.debug(f'New objects on disk: {len(objects_on_disk)}, last Modified: {last_modified}')

            if galaxy['specObjID'] in objects_on_disk:
                self.logger.debug(f'Preprocessed image for galaxy with id {galaxy["specObjID"]} already saved, skipping')
                continue
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
                    assert Path(file.replace('.bz2', '')).exists(), "Uncompressed file doesn't exist"
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
            redshift_objects.put(dict(key=galaxy['specObjID'],
                                      np_array=data_mat,
                                      redshift=galaxy['z'],
                                      galaxy_meta=galaxy,
                                      image=image_filename,
                                      timestamp=datetime.now()))

            galaxy_count += 1
            counter.value += 1
            self.logger.info(f'Galaxy count: {counter.value}, steps till next checkpoint: '
                             f'{self.checkpoint_steps - (counter.value % self.checkpoint_steps)}')
            dl_count = 0
            if counter.value % self.checkpoint_steps == 0:
                self.save_checkpoint(objs=redshift_objects, guid=guid)
        return redshift_objects

    def save_checkpoint(self, objs, guid):
        """Update checkpoint to a new step by adding extra checkpoint objects"""
        ckpt = CheckPoint.from_checkpoint(self.checkpoint_dir, guid)
        obj_list = []
        while not objs.empty():
            obj_list.append(objs.get())
        ckpt.save_checkpoint(obj_list)
        self.logger.debug(f'Saved objects with keys: {[obj["key"] for obj in obj_list]}')

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
        """Get formatted urls for this galaxy

        This method takes in the `rerun` number, `run` number, `camcol` and `filed`
        for a sdss galaxy and returns a set of URLs for Science Archive Server (SAS)
        fits files for each SDSS band `u`, `g`, `r`, `i` and `z` for the galaxy.

        Parameters
        ----------
        rerun : int
            The rerun number from SDSS survey
        run : int
            The run number from SDSS survey
        camcol : int
            The camera column number from SDSS survey
        field : int
            The field number from SDSS survey
        """
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
        """Return hmsdms string from ra, dec"""
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
        """Run the swarp tool as a subprocess

        Parameters
        ----------
        fits_file : str
            The fits file name for applying swarp tool on
        config_file : str
            The swarp config file to use
        center : 2-tuple or list
            The center of the fits around with swarp resampling is performed
        image_out : str
            The temporary filename for saving `IMAGEOUT_NAME` for swarp
        weight_out : str
            The temporary filename for saving `WEIGHTOUT_NAME` for swarp
        Notes
        -----
            This method assumes that swarp is available in the PATH
        """
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
        """Return a multiprocessing pool

        Parameters
        ----------
        num: int
            Number of processes to return
        """
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
@click.option('--volume-name',
              default='',
              type=str,
              help='The sciserver-volume in which checkpoint directory should reside(should be in Storage pool)')
@click.option('--num-processes',
              default=10,
              type=int,
              help='The number of processes to use (uses multiprocessing.Pool)')
def main(**kwargs):
    cs = PreProcess(**kwargs)
    cs.run()


if __name__ == '__main__':
    main()
