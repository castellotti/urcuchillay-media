# utils.py
# Copyright (c) 2024 Steve Castellotti
# This file is part of Urcuchillay and is released under the Apache 2.0 License.
# See LICENSE file in the project root for full license information.

import argparse
import platform

import config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command parameters')
    parser = parse_arguments_common(parser)
    args = parser.parse_args()
    return args


def parse_arguments_common(parser):
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=config.Config.DEBUG,
                        help='Enable debug mode (default: %(default)s)')
    parser.add_argument('--data', '--data_path', type=str, default=config.Config.DATA_PATH,
                        help='The path to data files to be indexed (default: %(default)s)')
    parser.add_argument('--path', '--model_path', type=str, default=config.Config.MODEL_PATH,
                        help='The path to the directory for cached models (default: %(default)s)')
    return parser


def str2bool(arg):
    """Parse boolean arguments."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ('yes', 'true', 'on', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'off', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_mac():
    return platform.system() == 'Darwin'
