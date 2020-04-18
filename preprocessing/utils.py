import os
import shutil


def zip_dir(base_name, base_dir=None, format='tar', dest_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()
    base_path = dest_dir + '/' + base_name
    print(base_path)
    archive_name = shutil.make_archive(base_name=base_path,
                                       root_dir=base_dir,
                                       format=format,
                                       dry_run=False)
    return archive_name