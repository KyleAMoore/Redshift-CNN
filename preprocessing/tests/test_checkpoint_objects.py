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

    def test_checkpoint(self):
        kwargs = [{
            'np_array': np.random.rand(64, 64, 5),
            'redshift': random.random(),
            'galaxy_meta': {'name': 'this is my name'},
            'image': 'path/to/image',
            'timestamp': datetime.now()
        }] * 5
        TestCheckpoint.ckpt_object.save_checkpoint(kwargs, overwrite=True)

    def test_restore_checkpoint(self):
        restored = CheckPoint.from_checkpoint('ckptObject.ckpt')
        assert [isinstance(meta_obj, restored.metaclass) for meta_obj in restored.meta_objects] \
            == [True] * len(restored.meta_objects)

    @classmethod
    def tearDownClass(cls):
        os.remove('ckptObject.ckpt')


if __name__ == '__main__':
    unittest.main()
