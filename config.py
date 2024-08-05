#!/usr/bin/env python3
# Copyright (c) 2024 Steve Castellotti
# This file is part of Urcuchillay and is released under the Apache 2.0 License.
# See LICENSE file in the project root for full license information.

import json
import logging
import os


class Config:
    DEBUG = False

    LOG_LEVELS = ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG', 'NOTSET']
    LOG_LEVEL = logging.DEBUG if DEBUG else logging.ERROR

    CONFIG_FILE = 'config.json'

    DATA_PATH = 'data'
    MODEL_PATH = 'models'
    COLLECTION_NAME = "urcuchillay"

    ENVIRONMENT = 'local'

    if ENVIRONMENT == 'local':
        MILVUS_URI = os.path.join(os.path.expanduser('~'), '.radient', 'default.db')
    elif ENVIRONMENT == 'container':
        MILVUS_URI = os.path.join(DATA_PATH, 'default.db')
    elif ENVIRONMENT == 'service':
        MILVUS_URI = "http://127.0.0.1:19530"

    TOKENIZERS_PARALLELISM = False


def load_config():
    try:
        with open(Config.CONFIG_FILE, 'r') as f:
            config_data = json.load(f)

        for key, value in config_data.get('Config', {}).items():
            setattr(Config, key, value)

    except FileNotFoundError:
        print('Config file not found, using default settings.')
    except PermissionError:
        print('Config file permission denied, using default settings.')
    except json.JSONDecodeError:
        print('Error decoding JSON, using default settings.')


load_config()
