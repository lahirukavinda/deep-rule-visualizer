#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv

from utils.arg_parse import *
from utils.dictionaries import *
from xDNN.Feature_Extraction_VGG16 import *
from xDNN.xDNN_run import *
from data_processing.saving_mnist_as_jpg_images import *

if __name__ == '__main__':
    args = get_args(data_set_dict)
    load_dotenv()  # load the list of local env variables from .env file

    data_set = args.data
    data_set_dir = os.getenv(data_set_dict[data_set])

    if not os.path.exists(f'features_{data_set}'):
        print(f'Extracting features from {data_set} data set...')
        os.mkdir(f'features_{data_set}')
        if data_set == 'mnist':
            download_mnist()
            VGG16_feature_extract_mnist(data_set, data_set_dir)
        else:
            VGG16_feature_extract(data_set, data_set_dir)
        print(f'Features were stored in features/{data_set} directory.\n')
    else:
        print(f'Features are already available in features/{data_set} directory.\n')

    xDNN_run(data_set)
