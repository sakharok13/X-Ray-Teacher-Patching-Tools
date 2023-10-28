import os
import datetime
import logging

from logging.handlers import RotatingFileHandler


def create_root_handler() -> logging.Handler:
    log_formatter = logging.Formatter('[%(levelname)s][%(asctime)s](%(funcName)s:%(lineno)d) %(message)s')
    log_formatter.datefmt = '%m.%d.%Y %H:%M:%S'

    logs_dir = os.path.join('.', 'temp')
    os.makedirs(logs_dir, exist_ok=True)

    current_timestamp = datetime.datetime.now().strftime("%d.%m.%y_%H%M")

    handler = RotatingFileHandler(filename=os.path.join(logs_dir, f"patching_{current_timestamp}.logs"),
                                  mode='a',
                                  maxBytes=5 * 1024 * 1024,  # 5Mb
                                  backupCount=10,
                                  encoding='utf-8')

    handler.setFormatter(log_formatter)

    return handler
