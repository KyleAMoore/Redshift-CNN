import os
import random
import unittest
from datetime import datetime

import numpy as np

from preprocessing.sdss.checkpoint_objects import CheckPoint, RedShiftCheckPointObject


class TestCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_object = CheckPoint(os.getcwd(), RedShiftCheckPointObject, guid='ckptObject')
        kwargs = [{
            'key': i,
            'np_array': np.random.rand(64, 64, 5),
            'redshift': random.random(),
            'galaxy_meta': {'name': 'this is my name'},
            'image': 'path/to/image',
            'timestamp': datetime.now()
        } for i in range(5)]
        TestCheckpoint.ckpt_object.save_checkpoint(kwargs, overwrite=True)

    def test_meta_objects(self):
        restored = CheckPoint.from_checkpoint(os.getcwd(), 'ckptObject')
        for meta_object in restored.meta_objects.values():
            assert type(meta_object) == RedShiftCheckPointObject
            assert isinstance(meta_object.key, int)
            assert isinstance(meta_object.np_array, np.ndarray)
            assert isinstance(meta_object.galaxy_meta, dict)
            assert isinstance(meta_object.image, str)
            assert isinstance(meta_object.timestamp, datetime)

    def test_checkpoint_exists(self):
        restored = CheckPoint.get_object_set(os.getcwd(), 'ckptObject')
        assert 0 in restored
        assert 1 in restored
        assert 2 in restored
        assert 3 in restored
        assert 4 in restored

    def test_add_objects(self):
        restored = CheckPoint.from_checkpoint(os.getcwd(), 'ckptObject')
        assert len(restored.meta_objects) == 5
        kwargs = [{
            'key': i,
            'np_array': np.random.rand(64, 64, 5),
            'redshift': random.random(),
            'galaxy_meta': {'name': 'this is my name'},
            'image': 'path/to/image',
            'timestamp': datetime.now()
        } for i in range(4, 10)]
        TestCheckpoint.ckpt_object.save_checkpoint(kwargs, 'ckptObject')
        restored_set = CheckPoint.get_object_set(os.getcwd(), 'ckptObject')
        assert len(restored_set) == 10

    def test_restore_checkpoint(self):
        restored = CheckPoint.from_checkpoint(os.getcwd(), 'ckptObject')
        assert [isinstance(meta_obj, restored.metaclass) for meta_obj in restored.meta_objects.values()] \
            == [True] * len(restored.meta_objects)

    @classmethod
    def tearDownClass(cls):
        CheckPoint.remove_ckpt(os.getcwd(), 'ckptObject')


if __name__ == '__main__':
    unittest.main()
