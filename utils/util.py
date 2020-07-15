#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import numpy as np


def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    # 定义输出log的格式
    logging.basicConfig(filename=log_file_path,
                        format='%(asctime)s %(levelname)-8s %(filename)s[line:%(lineno)d]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s[line:%(lineno)d]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    logger = logging.getLogger('PHOC_FPN')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger
