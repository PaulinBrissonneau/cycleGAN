#reader of the config file

import json

"""
Les valeurs par défauts si besoin :

CONFIG['load_epoch'] = False
CONFIG['end_epoch'] = 100
CONFIG['batch_size'] = 1
CONFIG['test_ratio'] = 0.2
CONFIG['vis_lines'] = 1
CONFIG['vis_rows'] = 3
CONFIG['plot_size'] = 20
CONFIG['alpha'] = 0.0002
CONFIG['beta_1'] = 0.5
CONFIG['dataset'] = "vangogh2photo"
CONFIG['max_buffer_size'] = 50
"""


def read_config (config_file):
    #in : config file (str)
    #out : dico (all params)

    with open('commun_cycleGAN/'+config_file) as config_file:
        CONFIG = json.loads(config_file.read())

    
    return CONFIG