
import copy
import functools
import os
import re
import yaml

def gather_args(args, name, force_existence=True):
    values = []
    for i, arg in enumerate(args):
        if arg == name:
            values.append( args[i+1] )
    
    assert len(values) > 0 or not force_existence, f'No {name} argument found. '
    
    return values

def remove_from_args_make_copy(args, arg_name_dict):
    '''
    arg_name_dict is a dictionary that maps the name of an argument to the number of values after
    that arguement. It assumes arguments with the same name have the same number of values. Number
    of values could be zero.
    '''
    flags = []
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in arg_name_dict:
            for _ in range( arg_name_dict[arg] + 1 ):
                flags.append(False)
            
            i += arg_name_dict[arg] + 1
        else:
            flags.append(True)
            i += 1
        
    return [ arg for arg, flag in zip(args, flags) if flag ]

def separate_dict_key_chain_prefix(s):
    m = re.search(r'^([\w\d\.]+)\@.+\.yaml$', s)
    if m:
        return m.group(1), s[len(m.group(1))+1:]
    else:
        return None, s

def substitute_config_yaml(d):
    for k, v in d.items():
        if isinstance(v, str) and v.lower().endswith(".yaml"):
            d[k] = read_config(v, recursive=True)
        elif isinstance(v, dict):
            substitute_config_yaml(v)

def read_config(fn, recursive=False):
    '''Complex config file reader.
    
    This reader deals with file names that could have a prefix in the form of 
    "key0.key1@config.yaml". When this happens, the reader will first read the yaml file and then
    get the value of config[key0][key1] and return it. When there is not a prefix (not a learding 
    string that ends with "@"), then the config file is read as usual.
    
    When recursive is True, then this reader will recursively read any value that is a string and 
    ends with ".yaml". The above prefix rule also applies.
    
    Returns:
    A dictionary that represnet the content of the config file.
    '''
    # Check if there is a prefix in the filename.
    prefix, fn = separate_dict_key_chain_prefix(fn)
    
    with open(fn, "r") as fp:
        # param_dict = yaml.safe_load(fp)
        param_dict = yaml.load(fp, Loader=yaml.FullLoader)

    if recursive:
        for k, v in param_dict.items():
            if isinstance(v, str) and v.lower().endswith(".yaml"):
                param_dict[k] = read_config(v, recursive=True)
            elif isinstance(v, dict):
                substitute_config_yaml(v)
                
    # If we need to get the special value from the key chain.
    if prefix is not None:
        key_chain = prefix.split(".")
        value = param_dict
        for k in key_chain:
            value = value[k]
        param_dict = value
    
    return param_dict

def merge_dicts( d_to, d_from, path=[] ):
    '''
    Inspired by https://gist.github.com/angstwad/bf22d1822c38a92ec0a9 and
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries?page=1&tab=scoredesc#tab-top
    
    Merge from d_from to d_to. If there is a key conflict, then d_from's value will be used and a
    message will be printed to the terminal.
    
    d_to is updated in-place.
    '''
    
    for key, value in d_from.items():
        if ( key in d_to ) and isinstance(value, dict) and isinstance(d_to[key], dict):
            merge_dicts(d_to[key], value, path + [str(key)])
        else:
            if key in d_to:
                print(f'Key override at {".".join(path + [str(key)])}. ')

            d_to[key] = value
            
    return d_to

def construct_dict_from_dot_notation(key, value):
    '''key is a string of the form "a.b.c". This function will return a dictionary with the
    following structure:
    {
        "a": {
            "b": {
                "c": value
            }
        }
    }
    '''
    keys = key.split(".")
    keys.reverse()
    v = value
    for k in keys:
        d = dict()
        d[k] = v
        v = d
    return d

def substitute_sweep_at(d, sweep_at_dict, prefix='sweep@'):
    for k, v in d.items():
        if isinstance(v, str) and v.startswith(prefix):
            d[k] = sweep_at_dict[v]
        elif isinstance(v, dict):
            substitute_sweep_at(v, sweep_at_dict, prefix=prefix)

def construct_config_on_filesystem(
        argv,
        top_level_single_key=None,
        name_config_base='--config_base', 
        name_config_sweep='--config_sweep',
        name_config_name='--config_name',
        **kwargs ):
    '''
    This function reads the values in argv.
    
    It is assumed that there is at least a name_config_base argument and/or a name_config_sweep argument. 
    If not, then this function does nothing and returns None. If the condition is met, then this
    function tries to combine all the name_config_sweep into a single dictionary and all the 
    name_config_base will be combined into a single dictionary. Then the sweep dictionary will override
    the values in the base dictionary. The final dictionary will be saved as a yaml file. The name 
    of the yaml file will be determined by the name_config_name argument, which is assumed to exist.
    
    Returns:
    The path to the generated yaml file.
    '''
    
    # Gather all the --config_base arguments.
    config_base_paths = gather_args(argv, name_config_base)
    config_bases = [ read_config(p) for p in config_base_paths ]
    config_base = functools.reduce(merge_dicts, config_bases)
    
    # Gather all the --config_sweep arguments.
    config_sweep_paths = gather_args(argv, name_config_sweep)
    config_sweeps = [ read_config(p) for p in config_sweep_paths ]
    config_sweep = functools.reduce(merge_dicts, config_sweeps)
    
    config_sweep_as_list_of_dicts = [ 
        construct_dict_from_dot_notation(k, v) for k, v in config_sweep.items() ]
    
    config_sweep = functools.reduce(merge_dicts, config_sweep_as_list_of_dicts)
    
    # Split the sweep@ entries in config_sweep.
    sweep_at_entries = dict()
    to_pop = list()
    for k, v in config_sweep.items():
        if k.startswith("sweep@"):
            sweep_at_entries[k] = v
            to_pop.append(k)
    
    # Remove the sweep@ entries from config_sweep.
    for k in to_pop:
        config_sweep.pop(k)
    
    # Merge the base and sweep dictionaries.
    config = copy.deepcopy(config_base)
    merge_dicts(config, config_sweep)
    
    # Perform YAML file substitution.
    substitute_config_yaml(config)
    
    # Perform sweep@ substitution.
    substitute_sweep_at(config, sweep_at_entries)
    
    # Get the top level single key.
    if top_level_single_key is not None:
        config = config[top_level_single_key]
    
    # Add additional key-value pairs.
    for k, v in kwargs.items():
        config[k] = v
    
    # Save the config to a file.
    config_names = gather_args(argv, name_config_name)
    
    if len(config_names) > 1:
        error_msg = f'Only one {name_config_name} argument is allowed. There are {len(config_names)} found: \n'
        for config_name in config_names:
            error_msg += f'\t{config_name}\n'
        raise ValueError(error_msg)
    elif len(config_names) == 0:
        raise(f'No {name_config_name} argument found. ')
    
    config_name = config_names[0]
    
    os.makedirs(os.path.dirname(config_name), exist_ok=True)
    with open(config_name, "w") as fp:
        yaml.dump(config, fp)
        
    return config_name
        
if __name__ == '__main__':
    import sys
    config_name = construct_config_on_filesystem(sys.argv)
    config = read_config(config_name)
    