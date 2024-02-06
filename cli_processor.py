
from typing import Any, List

from jsonargparse import ArgumentParser
import os

from lightning import Trainer

from config_helper import ( 
        construct_config_on_filesystem,
        remove_from_args_make_copy )

# File level constants.
DEFAULT_CHECKPOINT_FN = '__NONE__'

def preprocess_args(argv: List[str], top_level_single_key=None) -> str:
    '''Preprocess command line arguments. argv, the list of arguments goes through the following
    processing steps:
    1. Values of the arguments --config_base, --config_sweep, and --config_name are used to
    construct a config file which is saved to the filesystem. 
    2. The above arguments are removed from argv and saved to a new argument list.
    3. The filename of the above generated config file is appended to the new argument list as
    the --config argument.
    4. The new argument list is returned as a list of strings.

    Arguments:
    argv: The list of arguments to be processed.

    Returns:
    A list of processed arguments as list of strings.
    '''

    # Local constants.
    name_config_base  = '--config_base'
    name_config_sweep = '--config_sweep'
    name_config_name  = '--config_name'
    
    config_name = construct_config_on_filesystem(
        argv,
        top_level_single_key=top_level_single_key,
        name_config_base=name_config_base,
        name_config_sweep=name_config_sweep,
        name_config_name=name_config_name )
    
    new_args = remove_from_args_make_copy(argv, 
            { name_config_base:  1, 
              name_config_sweep: 1,
              name_config_name:  1 } )
    
    # Append a --config argument to the list of arguments.
    new_args.append( '--config' )
    new_args.append( config_name )
    
    return new_args

class AdditionalArg(object):
    def __init__(self, cmd: str, type: Any, default: Any=None, help: str=''):
        super().__init__()
        self.cmd = cmd
        self.type = type
        self.default = default
        self.help = help

def parse_args_round_1(argv: List[str], additional_args: List[AdditionalArg]=None):
    '''Use jsonargparse to parse the arguments that might have been preprocessed by 
    preprocess_args().

    additional_args: A list of dictionary of additional arguments. Each dict has the following keys:
    cmd: The command line prefix, e.g. "--checkpoint_run_id".
    type: The type of the argument, e.g. str.
    default: The default value.
    help: The help message.

    This function returns the results from ArgumentParser.parse_args() defined by the 
    jsonargparse package.
    '''
    # global DEFAULT_CHECKPOINT_FN

    parser = ArgumentParser()
    parser.add_argument('--port_modifier', type=int, default=0, 
                        help='An integer for settingg an unique port number such that '
                             'multiple training/validation processes can run on the '
                             'same machine.')
    # parser.add_argument('--checkpoint_fn', type=str, default=DEFAULT_CHECKPOINT_FN,
    #                     help='The filename of the checkpoint. Do not specify this if the use '
    #                          'does not want to load a checkpoint.')
    parser.add_argument('--config', type=str, required=True,
                        help='The filename of the config file. Note that this may be generated '
                             'on the fly.')
    
    if additional_args is not None:
        for arg in additional_args:
            parser.add_argument(arg.cmd, type=arg.type, default=arg.default, help=arg.help)
    
    return parser.parse_args(args=argv)

def parse_args_from_config(fn: str, instantiate_classes: bool=True):
    '''This function applies the magic of jsonargparse to instantiate class objects from a config
    file. Currently only the following keys are parsed as class objects: trainer, data, model, 
    and optimizer. 

    This function returns the instantiated class objects as well as the raw arguments as a 
    dictionary. 

    Note that jsonargparse may fail and present the user with an error message like the 
    following: TODO: add error message sample.

    This is typically due to a wrong class name or a wrong argument used for instantiating a 
    class object. It could be very difficult to debug. All of our example config have been 
    tested. So a more safe way to create a new config file is first copy an existing example and
    modify it fo the user's need.
    '''
    parser = ArgumentParser()
    
    parser.add_class_arguments(Trainer, 'trainer')
    parser.add_argument('--data', type=Any)
    parser.add_argument('--model', type=Any)
    parser.add_argument('--optimizer', type=Any)
    
    raw_cfg = parser.parse_path(fn)
    cfg = parser.instantiate_classes(raw_cfg) \
        if instantiate_classes \
        else None

    return cfg, raw_cfg

def set_master_port(port_modifier: int):
    '''65535 is the max port number supported by our Linux system. This function generates a 
    5 digit port number that starts with 20. port_modifier % 1000 is first computed internally. 
    '''
    port_modifier = port_modifier % 1000
    os.environ['MASTER_PORT'] = f'20{port_modifier:03d}'
