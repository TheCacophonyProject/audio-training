"""

Author Giampaolo Ferraro

Date December 2023

Handles reading and writing tracks (or segments) to a large database.  Uses HDF5 as a backing store.

"""
import h5py
import os
import logging
import filelock
import numpy as np
from dateutil.parser import parse as parse_date
import json

import numpy as np

special_datasets = [
    "tag_frames",
    "original_frames",
    "background_frame",
    "predictions",
    "overlay",
]


class HDF5Manager:
    """Class to handle locking of HDF5 files."""

    LOCK_FILE = "/var/lock/classifier-hdf5.lock"
    READ_ONLY = False

    def __init__(self, db, mode="r"):
        self.mode = mode
        self.f = None
        self.db = db
        self.lock = filelock.FileLock(HDF5Manager.LOCK_FILE, timeout=60 * 3)
        filelock.logger().setLevel(logging.ERROR)

    def __enter__(self):
        # note: we might not have to lock when in read only mode?
        # this could improve performance
        if HDF5Manager.READ_ONLY and self.mode != "r":
            raise ValueError("Only read can be done in readonly mode")
        if not HDF5Manager.READ_ONLY:
            self.lock.acquire()
        self.f = h5py.File(self.db, self.mode)
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.f.close()
        finally:
            if not HDF5Manager.READ_ONLY:
                self.lock.release()


class AudioDatabase:
    def __init__(self, database_filename, read_only=False):
        """
        Initialises given database.  If database does not exist an empty one is created.
        :param database_filename: filename of database
        """

        self.database = database_filename
        if not os.path.exists(database_filename):
            logging.info("Creating new database %s", database_filename)
            f = h5py.File(database_filename, "w")
            f.create_group("recs")
            f.close()
        HDF5Manager.READ_ONLY = read_only

    def set_read_only(self, read_only):
        HDF5Manager.READ_ONLY = read_only

    def has_rec(self, rec_id):
        """
        Returns if database contains track information for given clip
        :param clip_id: name of clip
        :return: If the database contains given clip
        """
        with HDF5Manager(self.database) as f:
            clips = f["recs"]
            has_record = rec_id in clips and "finished" in clips[rec_id].attrs
            if has_record:
                return True
        return False
