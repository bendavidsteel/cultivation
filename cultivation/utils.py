import math

import numpy as np

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, manager, file_path, cb_name):
        self.manager = manager
        self.file_path = file_path
        self.cb_name = cb_name

    def on_modified(self, event):
        if event.src_path == self.file_path:
            cb = getattr(self.manager, self.cb_name)
            cb()

def setup_observer(file_path, manager, cb_name):
    assert hasattr(manager, cb_name), f"Manager does not have a method {cb_name}"

    event_handler = FileChangeHandler(manager, file_path, cb_name)

    observer = Observer()
    observer.schedule(event_handler, path=file_path, recursive=False)
    observer.start()

    return event_handler, observer

def eval_statement(r, kwargs, default, expression_name, logger):
    try:
        r = eval(r, {}, {'fft': kwargs['fft'], 'time': kwargs['time'], 'np': np, 'math': math, 'sin': np.sin, 'cos': np.cos})
    except Exception as e:
        logger.error(f"Error evaluating {expression_name} expression: {e}")
        r = default
    return r