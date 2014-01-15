import datetime
import sys
import os

def make_plot_filename(main_title, directory='.', config_dict={}, use_date=True, sep='_'):
    """
    param main_title: the main title for the file
    type string:

    config_dict: maps labels to values
    """

    labels = [main_title]
    for label, value in config_dict.iteritems():
        labels.append(label)
        labels.append(str(value))

    if use_date:
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":", " ","-"])
        labels.append(date_time_string)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    return sep.join(labels)







