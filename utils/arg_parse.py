import argparse


def get_args(data_set_dict):
    parser = argparse.ArgumentParser()

    data_set_choices = list(data_set_dict.keys())
    parser.add_argument('-d', '--data', choices=data_set_choices, help=f'Select the dataset from {data_set_choices}')
    args = parser.parse_args()

    return args
