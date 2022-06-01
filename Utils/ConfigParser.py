import yaml

import pprint
import argparse

from enum import Enum

# TODO: Why using them???
class FileSelectionMode(Enum):
    RANDOM,     \
    SHUFFLE,    \
    DEFAULT     \
    = range(3)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return FileSelectionMode[s]
        except KeyError:
            raise ValueError()
class SampleExtractionMode(Enum):
    RANDOM,     \
    SEQUENTIAL  \
    = range(2)

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return SampleExtractionMode[s]
        except KeyError:
            raise ValueError()

KEY_BACKGROUND_LAYER = "rgba PNG - Layer 0 (Background)"
KEY_SELECTED_REGIONS = "rgba PNG - Selected regions"
KEY_RESOURCE_PATH = "resource_path"

kPATH_IMAGES_DEFAULT = "datasets/images"
kPATH_REGION_MASKS_DEFAULT = "datasets/regions"
kPATH_BACKGROUND_DEFAULT = "datasets/layers/background"
kPATH_LAYERS_DEFAULT = ["datasets/layers/staff", "datasets/layers/neumes"]
kPATH_OUTPUT_MODELS_DEFAULT = ["Models/model_background.h5", "Models/model_staff.h5", "Models/model_neumes.h5"]
kBATCH_SIZE_DEFAULT = 8
kPATCH_HEIGHT_DEFAULT = 256
kPATCH_WIDTH_DEFAULT = 256
kMAX_NUMBER_OF_EPOCHS_DEFAULT = 1
kNUMBER_SAMPLES_PER_CLASS_DEFAULT = 100
kEARLY_STOPPING_PATIENCE_DEFAULT = 15
kFILE_SELECTION_MODE_DEFAULT = FileSelectionMode.SHUFFLE
kSAMPLE_EXTRACTION_MODE_DEFAULT = SampleExtractionMode.RANDOM

# Unused function that parse command line arguments 
def parseArgs():
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description='Fast trainer')
    # Original Image
    parser.add_argument( '-psr', default=kPATH_IMAGES_DEFAULT, 
                    dest='path_src', 
                    help='Path of the source folder that contains the original images.'
                    )
    # Region
    parser.add_argument('-prg', default=kPATH_REGION_MASKS_DEFAULT,
                    dest='path_regions', 
                    help='Path of the folder that contains the region masks.'
                    )
    # Background
    parser.add_argument('-pbg', default=kPATH_BACKGROUND_DEFAULT,
                    dest='path_bg', 
                    help='Path of the folder with the background layer data.'
                    )
    # Layers list
    parser.add_argument('-pgt', dest='path_layer', 
                    help='Paths of the ground-truth folders to be considered (one per layer).', 
                    action='append'
                    )
    # Output list
    parser.add_argument('-out', dest='path_out', 
                    help='Paths for the models saved after the training.', 
                    action='append'
                    )
    # Patch width
    parser.add_argument('-width', default=kPATCH_HEIGHT_DEFAULT,
                    dest='patch_width',
                    type=int,
                    help='Patch width'
                    )
    # Patch height
    parser.add_argument('-height', default=kPATCH_WIDTH_DEFAULT,
                    dest='patch_height',
                    type=int,
                    help='Patch height'
                    )
    # Batch size
    parser.add_argument('-b', default=kBATCH_SIZE_DEFAULT,
                    dest='batch_size',
                    type=int,
                    help='Batch size'
                    )
    # Epoch
    parser.add_argument('-e', default=kMAX_NUMBER_OF_EPOCHS_DEFAULT,
                    dest='max_epochs',
                    type=int,
                    help='Maximum number of epochs'
                    )
    # The number of patches per epoch
    parser.add_argument('-n', default=kNUMBER_SAMPLES_PER_CLASS_DEFAULT,
                    dest='number_samples_per_class',
                    type=int,
                    help='Number of samples per class to be extracted'
                    )
    # Mode to select file (shuffle?)
    parser.add_argument('-fm', default=kFILE_SELECTION_MODE_DEFAULT, 
                    dest='file_selection_mode',
                    type=FileSelectionMode.from_string, 
                    choices=list(FileSelectionMode), 
                    help='Mode of selecting images in the training process'
                    )
    # Mode to extract patches
    parser.add_argument('-sm', default=kSAMPLE_EXTRACTION_MODE_DEFAULT, 
                    dest='sample_extraction_mode',
                    type=SampleExtractionMode.from_string, 
                    choices=list(SampleExtractionMode), 
                    help='Mode of extracing samples for each image in the training process'
                    )
    # Early stop
    parser.add_argument('-pat', default=kEARLY_STOPPING_PATIENCE_DEFAULT,
                    dest='patience',
                    type=int,
                    help='Number of epochs of patience for the early stopping. If the model does not improves the training results in this number of consecutive epochs, the training is stopped.'
                    )

    args = parser.parse_args()

    args.path_layer = args.path_layer if args.path_layer is not None else kPATH_LAYERS_DEFAULT
    args.path_out = args.path_out if args.path_out is not None else kPATH_OUTPUT_MODELS_DEFAULT

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def getDefaultConfig():
    """Return default configuration.

    When using yaml, default config is just a dictionary.
    """
    tmp = {   'batch_size': kBATCH_SIZE_DEFAULT,
    'max_epochs': kBATCH_SIZE_DEFAULT,
    'number_samples_per_class': kNUMBER_SAMPLES_PER_CLASS_DEFAULT,
    'patch_height': kPATCH_HEIGHT_DEFAULT,
    'patch_width': kPATCH_WIDTH_DEFAULT,
    'path_bg': kPATH_BACKGROUND_DEFAULT,
    'path_layer': kPATH_LAYERS_DEFAULT,
    'path_out': kPATH_OUTPUT_MODELS_DEFAULT,
    'path_regions': kPATH_REGION_MASKS_DEFAULT,
    'path_src': kPATH_IMAGES_DEFAULT,
    'patience': kEARLY_STOPPING_PATIENCE_DEFAULT,
    'sample_extraction_mode': 'RANDOM',
    'file_selection_mode': 'SHUFFLE'}
    return tmp

def loadConfig(config_path:str):
    """Read config from yaml file.
    
    Read the yaml file and merge it with the default yaml. Return a argparse.Namespace.
    YAML is a good stuff, please use yaml, don't use argparse they are evil!

    Parameters:
        config_path (str): Input yaml path.
    Returns:
        argparse.Namespace : This is a namespace. (?)
    """

    # Load default config
    config = getDefaultConfig()

    # Read from yaml file
    with open(config_path, "r") as fp:
        user_config = yaml.safe_load(fp)
    config.update(user_config)

    # TODO: Read from command line

    # TODO: Remove these weird enum classes!
    config = argparse.Namespace(**config)
    config.sample_extraction_mode = SampleExtractionMode.from_string(config.sample_extraction_mode)
    config.file_selection_mode = FileSelectionMode.from_string(config.file_selection_mode)

    return config