import pickle
import os
from datetime import datetime

from dataclasses import dataclass

import numpy as np

from logger_factory import LoggerFactory


@dataclass
class RedShiftCheckPointObject:
    """Checkpoint object for saving redshift data for galaxies"""
    key: str
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
    meta_objects: dict, default=None
        a dictionary containing metaobjects' key as key and meta objects as values
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
        self.meta_keys = set(self.meta_objects.keys())

    def _validate_metaobjects(self, meta_objects):
        """Validate `meta_objects` as instances of the metaclass"""
        if meta_objects is None:
            return {}
        else:
            for obj in meta_objects.values():
                assert isinstance(obj, self.metaclass), \
                    'Invalid object of type {}, for class {}'.format(type(obj), self.metaclass)
            return meta_objects

    def save_checkpoint(self, obj_kwargs_list=None, overwrite=True):
        """Save the checkpoint object

        This function saves the checkpoint object as a pkl file in the checkpoint directory

        Arguments:
        ----------
        obj_kwargs : list of dict
            A list of kwargs to self.meta_class
        overwrite : bool, default=True
            If True, this will remove the old objects if they exist in the checkpointer
        """
        if obj_kwargs_list is not None:

            for obj_kwargs in obj_kwargs_list:
                if overwrite:
                    self.meta_keys.discard(obj_kwargs['key'])
                    exists = self.meta_objects.pop(obj_kwargs['key'], None)
                    if exists:
                        self.logger.debug(f'Key {obj_kwargs["key"]} already exists in the checkpoint. Overwriting')
                if obj_kwargs['key'] not in self.meta_keys:
                    checkpoint_object = self.metaclass(**obj_kwargs)
                    self.meta_objects[obj_kwargs['key']] = checkpoint_object
                    self.meta_keys.add(obj_kwargs['key'])
                    self.logger.debug(f'Added checkpoint object with key {obj_kwargs["key"]} to the checkpoint')

        with open(os.path.join(self.checkpoint_dir, self.guid + '.ckpt'), 'wb') as pklfile:
            pickle.dump(self, pklfile)
            self.logger.debug(f'Saved checkpoint with {len(self.meta_keys)} objects '
                              f'as {os.path.join(self.checkpoint_dir, self.guid + ".ckpt")}')

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

    @staticmethod
    def get_object_set(ckptdir, guid):
        """Return the set of object keys in the current checkpoint(on disk)

        Parameters
        ----------
        ckptdir : str or os.path like
            Checkpoint directory to look in
        guid: str
            GUID for the checkpoint

        Returns
        -------
        set
            A set of meta object keys for this checkpoint (if exists)
        """
        checkpoint = os.path.join(ckptdir, f'{guid}.ckpt')
        if not os.path.exists(checkpoint):
            return set()
        else:
            ckpt = CheckPoint.from_checkpoint(ckptdir, guid)
            return ckpt.meta_keys

    @classmethod
    def from_checkpoint(cls, ckptdir, guid):
        """Return checkpoint object from the pickle file"""
        assert CheckPoint.checkpoint_exists(ckptdir, guid)
        obj_pkl = os.path.join(ckptdir, guid + '.ckpt')
        with open(obj_pkl, 'rb+') as obj:
            checkpoint_obj = pickle.load(obj)
        assert isinstance(checkpoint_obj, CheckPoint), type(checkpoint_obj)
        checkpoint_obj.logger.debug(f'Restored checkpoint with guid {checkpoint_obj.guid}, '
                                    f'number of objects: {len(checkpoint_obj.meta_keys)}')
        return checkpoint_obj

    @staticmethod
    def last_modified(ckptdir, guid):
        assert CheckPoint.checkpoint_exists(ckptdir, guid)
        return os.path.getmtime(os.path.join(ckptdir, f'{guid}.ckpt'))