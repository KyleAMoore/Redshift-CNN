import pickle
import os
from datetime import datetime

from dataclasses import dataclass

import numpy as np


@dataclass
class RedShiftCheckPointObject:
    np_array: np.ndarray
    redshift: float
    galaxy_meta: dict
    image: str
    timestamp: datetime


class CheckPoint(object):
    """A generic checkpoint Object"""
    def __init__(self, checkpoint_dir, meta_class, guid, meta_objects=None):
        self.created_at = datetime.now()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir) and not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.metaclass = meta_class
        self.guid = guid
        self.logger = LoggerFactory.get_logger(self.__class__.__name__,
                                               'DEBUG')
        self.meta_objects = self._validate_metaobjects(meta_objects)

    def _validate_metaobjects(self, meta_objects):
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
        if os.path.exists(os.path.join(ckptdir, guid + '.ckpt')):
            return True
        return False

    @staticmethod
    def remove_ckpt(ckptdir, guid, throw=False):
        filename = os.path.join(ckptdir, guid + '.ckpt')
        if CheckPoint.checkpoint_exists(ckptdir, guid):
            os.remove(filename)
        else:
            if throw:
                raise FileNotFoundError('Checkpoint file {} not found'.format(filename))

    @classmethod
    def from_checkpoint(cls, obj_pkl):
        with open(obj_pkl, 'rb+') as obj:
            checkpoint_obj = pickle.load(obj)
        # assert isinstance(checkpoint_obj, .CheckPoint), type(checkpoint_obj)
        print(type(checkpoint_obj))
        return checkpoint_obj


if __name__ == '__main__':
    object_ = CheckPoint.from_checkpoint('/home/umesh/Downloads/8aa325d55bc2d254d8b8ccbf886ff9886446ab7c.ckpt')
    for redshift_obj in object_.metaclass:
        print(redshift_obj.np_array, redshift_obj.redshift)


