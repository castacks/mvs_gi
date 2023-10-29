
# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-11-01

import copy

DIST_LAB_TAB=dict() # Distance-label table.
GRID_MAKER_BUILDER=dict()
DATASET=dict()

def register(dst):
    '''Register a class to a dstination dictionary. '''
    def dec_register(cls):
        dst[cls.__name__] = cls
        return cls
    return dec_register

def make_support_object(typeD, argD, **additional):
    '''Make an object from type collection typeD. '''

    assert( isinstance(typeD, dict) ), f'typeD must be dict. typeD is {type(typeD)}'
    assert( isinstance(argD,  dict) ), f'argD must be dict. argD is {type(argD)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(argD)

    # Get the type.
    typeName = typeD[ d['type'] ]

    # Remove the type string from the input dictionary.
    d.pop('type')

    # Create the model.
    return typeName( **d, **additional )