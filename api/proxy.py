import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..')

if _TOP_PATH not in sys.path:
    sys.path.insert( 0, _TOP_PATH)
    for i, p in enumerate(sys.path):
        print(f'{i}: {p}')

from typing import Any, List
from jsonargparse import ArgumentParser

# import torch
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# Local packages.
# from dsta_mvs.model.globals import GLOBAL
import dsta_mvs
from dsta_mvs.mvs_utils import debug as mvs_debug
from dsta_mvs.mvs_utils.file_sys import get_filename_parts

from config_helper import ( 
        construct_config_on_filesystem,
        read_config,
        remove_from_args_make_copy )

class ProxyBase(object):
    # File level constants.
    DEFAULT_CHECKPOINT_FN = '__NONE__'

    def __init__(self, 
                 argv: List[str],
                 preprocessed_config = False,
                 debug=False
                 ):
        super().__init__()
        
        # Handle the global debug setting.
        if debug:
            mvs_debug.enable()
        else:
            mvs_debug.disable()

        # Parse the command line arguments.
        self.cfg = None
        self.raw_cfg = None
        if preprocessed_config:
            self.cfg, self.raw_cfg = self.parse_commandline_without_preprocessing(argv)
        else:
            self.cfg, self.raw_cfg = self.parse_commandline(argv)
        
        # # Handle global settings.
        # self.handle_global_settings()

    @staticmethod
    def convert_dict_2_argv(arg_dict):
        '''Convert a dictionary of arguments to a list of command line arguments. This is useful for
        passing arguments to a subprocess.

        Arguments:
        arg_dict: A dictionary of arguments.

        Returns:
        A list of command line arguments.
        '''
        argv = []
        for k, v in arg_dict.items():
            argv.append(f'--{k}')
            argv.append(f'{v}')
        return argv


    @staticmethod
    def find_checkpoint_from_argv(argv: List[str]):
        '''Find the checkpoint filename from the command line arguments.

        Arguments:
        argv: The list of command line arguments.

        Returns:
        The checkpoint filename.
        '''
        checkpoint_fn = ProxyBase.DEFAULT_CHECKPOINT_FN
        for i, arg in enumerate(argv):
            if arg == '--checkpoint_fn':
                checkpoint_fn = argv[i+1]
                break
        return checkpoint_fn

    # @staticmethod
    # def load_dynamic_conf(py_fn):
    #     import importlib.util
        
    #     sys_module_name = "mvs.dynamic_conf"
        
    #     spec = importlib.util.spec_from_file_location(sys_module_name, py_fn)
    #     dynamic_conf = importlib.util.module_from_spec(spec)
    #     sys.modules[sys_module_name] = dynamic_conf
    #     spec.loader.exec_module(dynamic_conf)
    #     return dynamic_conf.conf

    # @staticmethod
    # def load_json_conf(json_fn):
    #     with open(json_fn, 'r') as fp:
    #         return json.load(fp)

    # @staticmethod
    # def load_conf(fn):
    #     # Get the file extention of fn.
    #     parts = get_filename_parts(fn)
        
    #     if parts[-1] == ".py":
    #         return ProxyBase.load_dynamic_conf(fn)
    #     elif parts[-1] == ".json":
    #         return ProxyBase.load_json_conf(fn)
    #     else:
    #         raise Exception(f'Only accepts .py or .json as the extesion of the configuration file. fn = {fn}. ')

    @staticmethod
    def preprocess_args(argv: List[str]) -> str:
        '''Preprocess command line arguments. argv, the list of arguments goes through the following
        processing steps:
        1. Values of the arguments --config_base, --config_sweep, and --config_name are used to
        construct a config file which is saved to the filesystem. 
        2. The above arguments are removed from argv and saved to a new argument list.
        3. The filename of the above generated config file is appended to the new argument list as
        the --config argument.
        4. The new argument list is returned as a list of strings.

        The current implementation of this function assumes that there is a top-level "fit" key for 
        the YAML config file specified by the --config_base argument. This function only use the 
        contents under the "fit" key. An example of the YAML content of the --config_base argument 
        could be found at TODO: add link.

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
            top_level_single_key='fit',
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

    @staticmethod
    def parse_args_round_1(argv: List[str]):
        '''Use jsonargparse to parse the arguments that might have been preprocessed by 
        preprocess_args().

        This function returns the results from ArgumentParser.parse_args() defined by the 
        jsonargparse package.
        '''

        parser = ArgumentParser()
        parser.add_argument('--port_modifier', type=int, default=0, 
                            help='An integer for settingg an unique port number such that '
                                'multiple training/validation processes can run on the '
                                'same machine.')
        parser.add_argument('--checkpoint_fn', type=str, default=ProxyBase.DEFAULT_CHECKPOINT_FN,
                            help='The filename of the checkpoint. Do not specify this if the use '
                                'does not want to load a checkpoint.')
        parser.add_argument('--config', type=str, required=True,
                            help='The filename of the config file. Note that this may be generated '
                                'on the fly.')
        return parser.parse_args(args=argv)

    @staticmethod
    def parse_args_from_config(fn: str):
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
        
        # parser.add_class_arguments(Trainer, 'trainer')
        parser.add_argument('--data', type=Any)
        parser.add_argument('--model', type=dsta_mvs.model.mvs_model.spherical_sweep_stereo.SphericalSweepStereo)
        # parser.add_argument('--optimizer', type=Any)
        
        raw_cfg = parser.parse_path(fn)
        cfg = parser.instantiate_classes(raw_cfg)

        return cfg, raw_cfg

    @staticmethod
    def parse_commandline(argv: List[str]):
        argv = ProxyBase.preprocess_args(argv)
        args = ProxyBase.parse_args_round_1(argv)
        cfg, raw_cfg = ProxyBase.parse_args_from_config(args.config)
        return cfg, raw_cfg

    @staticmethod
    def parse_commandline_without_preprocessing(argv: List[str]):
        args = ProxyBase.parse_args_round_1(argv)
        cfg, raw_cfg = ProxyBase.parse_args_from_config(args.config)
        return cfg, raw_cfg

    # def handle_global_settings(self):
    #     config_globals = self.config['globals']
    #     GLOBAL.torch_align_corners(config_globals['align_corners'])
    #     GLOBAL.torch_align_corners_nearest(config_globals['align_corners_nearest'])
    #     GLOBAL.torch_batch_normal_track_stat(config_globals['track_running_stats'])
    #     GLOBAL.relu_type(config_globals['relu_type'])
    #     GLOBAL.torch_relu_inplace(config_globals['relu_inplace'])
