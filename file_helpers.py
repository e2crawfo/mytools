import datetime
import sys
import os
import string
import numpy as np

def make_filename(main_title, directory='.', config_dict={}, use_time=True, sep='_', extension='',
                  omit=[]):
    """
    :param main_title: the main title for the file
    :type string:

    :param directory: the directory to write the file to
    :type string:

    :param config_dict: maps labels to values
    :type dict:

    :param use_time: whether to append the current date/time to the filename
    :type bool:

    :param sep: string separating items in the config dict in the returned filename
    :param string:

    :param extension: string to appear at end of filename
    :param string:
    """

    if directory[-1] != '/': directory += '/'

    labels = [directory + main_title]
    for label, value in config_dict.iteritems():
        if not str(label) in omit and not hasattr(value, '__len__'):
            labels.append(str(label))
            labels.append(str(value))

    if use_time:
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":", " ","-"])
        labels.append(date_time_string)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    file_name = sep.join(labels)

    if extension:
        if extension[0] != '.': extension = '.' + extension
        file_name += extension

    return file_name

def npload(filename):
    data_loaded = True
    try:
        print "Trying to load..."
        f = open(filename, 'r')
        npfile = np.load(f)
        print "Loaded"
        return npfile
    except Exception as E:
        print E
        print "Couldn't load."
        return None

def npsave(filename, **kwargs):
    f = open(filename, 'w')
    np.savez(f, **kwargs)








