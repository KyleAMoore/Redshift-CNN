import pickle
import os
from datetime import datetime

from dataclasses import dataclass

import numpy as np

from logger_factory import LoggerFactory


@dataclass
class RedShiftCheckPointObject:
    """Checkpoint object for saving redshift data for galaxies"""
    np_array: np.ndarray
    redshift: float
    galaxy_meta: dict
    image: str
    timestamp: datetime


class CheckPoint(object):
    """A generic checkpointer class

    This class provides a basic checkpoint support for saving sdss images.
    The objects are saved to disk using pickle.

    Parameters
    ----------
    checkpoint_dir : str or Path like
        The directory to save the checkpoint in
    metaclass: instance of Object
        The class for which to save checkpoint objects for
    guid: str
        Global identifier for this checkpoint
    meta_objects: list or iterable, default=None
         a list of kwargs to `meta_class` constructor to create checkpoint objects
    """
    def __init__(self, checkpoint_dir, metaclass, guid, meta_objects=None):
        self.created_at = datetime.now()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir) and not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.metaclass = metaclass
        self.guid = guid
        self.logger = LoggerFactory.get_logger(self.__class__.__name__,
                                               'DEBUG')
        self.meta_objects = self._validate_metaobjects(meta_objects)

    def _validate_metaobjects(self, meta_objects):
        """Validate `meta_objects` as instances of the metaclass"""
        if meta_objects is None:
            return []
        else:
            for obj in meta_objects:
                assert isinstance(obj, self.metaclass), \
                    'Invalid object of type {}, for class {}'.format(type(obj), self.metaclass)
            return meta_objects

    def save_checkpoint(self, obj_kwargs_list=None, overwrite=True):
        """Save the checkpoint object

        This function saves the checkpoint object as a pkl file in
        Arguments:
        ----------
        obj_kwargs : list of dict
            A list of kwargs to self.meta_class
        overwrite : bool, default=True
            If True, this file will remove the old object if it exists
        """
        if obj_kwargs_list is not None:
            assert isinstance(obj_kwargs_list, list),\
                'Please provide a list of keyword arguments to your checkpoint object'
            for obj_kwargs in obj_kwargs_list:
                checkpoint_object = self.metaclass(**obj_kwargs)
                self.meta_objects.append(checkpoint_object)

        if os.path.exists(os.path.join(self.checkpoint_dir, self.guid + '.ckpt')):
            if overwrite:
                os.remove(os.path.join(self.checkpoint_dir, self.guid + '.ckpt'))
            else:
                return

        with open(os.path.join(self.checkpoint_dir, self.guid + '.ckpt'), 'wb') as pklfile:
            pickle.dump(self, pklfile)

    def get_loc(self):
        return os.path.join(self.checkpoint_dir, self.guid + '.ckpt')

    @staticmethod
    def checkpoint_exists(ckptdir, guid):
        """Check whether a checkpoint with given guid exists

        Parameters
        ----------
        ckptdir : str or os.path like
            The directory to check for checkpoints
        """
        if os.path.exists(os.path.join(ckptdir, guid + '.ckpt')):
            return True
        return False

    @staticmethod
    def remove_ckpt(ckptdir, guid, throw=False):
        """Remove the checkpoint with given guid

        Parameters
        ----------
        ckptdir : str or os.path like
            The directory to check for checkpoints
        guid : str
            The guid of the checkpoint
        throw: bool, default=False
            If true, throw an error if the checkpoint with given guid doesn't exist
        """
        filename = os.path.join(ckptdir, guid + '.ckpt')
        if CheckPoint.checkpoint_exists(ckptdir, guid):
            os.remove(filename)
        else:
            if throw:
                raise FileNotFoundError('Checkpoint file {} not found'.format(filename))

    @classmethod
    def from_checkpoint(cls, obj_pkl):
        """Return checkpoint object from the pickle file"""
        with open(obj_pkl, 'rb+') as obj:
            checkpoint_obj = pickle.load(obj)
        # assert isinstance(checkpoint_obj, .CheckPoint), type(checkpoint_obj)
        print(type(checkpoint_obj))
        return checkpoint_obj