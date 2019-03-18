import os
import yaml
import logging
import logging.config

_logger = logging.getLogger(__name__)
_logging_initialized = False


def init_env(env='dev'):
    init_logging()


def init_logging(cfg_path=None, level=logging.INFO):
    global _logging_initialized
    if not _logging_initialized:
        if cfg_path is not None and os.path.exists(cfg_path):
            with open(cfg_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] %(levelname)s %(name)s:%(lineno)d - %(message)s")
            rootLogger = logging.getLogger()
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            rootLogger.addHandler(consoleHandler)
            rootLogger.setLevel(level)
        _logging_initialized = True
