import logging
import colorlog


def get_logger(level=logging.INFO, file_path=None, mode='w', console=True):
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(level)
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # 定义颜色输出格式
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    # 将颜色输出格式添加到控制台日志处理器
    console_handler.setFormatter(color_formatter)
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # 将控制台日志处理器添加到logger对象
    if console:
        logger.addHandler(console_handler)
    file_formatter = colorlog.ColoredFormatter(
        '%(levelname)s - %(asctime)s - %(message)s',
    )
    f_handler = logging.FileHandler(file_path, mode=mode)
    f_handler.setFormatter(file_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)

    # Do not output logs in the console
    logger.propagate = False
    return logger
