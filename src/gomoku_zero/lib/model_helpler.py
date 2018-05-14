from logging import getLogger
from filelock import Timeout, FileLock

logger = getLogger(__name__)

lock_path = "file_lock.txt.lock"
lock = FileLock(lock_path, timeout=120)

def load_best_model_weight(model):
    """

    :param gomoku_zero.agent.model.ChessModel model:
    :return:
    """
    lock.acquire(timeout=120)
    m = model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)
    lock.release()
    return m

def load_baseline_model_weight(model):
    """

    :param gomoku_zero.agent.model.ChessModel model:
    :return:
    """
    lock.acquire(timeout=120)
    m = model.load(model.config.resource.model_baseline_config_path, model.config.resource.model_baseline_weight_path)
    lock.release()
    return m


def save_as_best_model(model):
    """

    :param gomoku_zero.agent.model.ChessModel model:
    :return:
    """
    lock.acquire(timeout=120)
    m = model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)
    lock.release()
    return m


def reload_best_model_weight_if_changed(model):
    """

    :param gomoku_zero.agent.model.ChessModel model:
    :return:
    """
    logger.debug(f"start reload the best model if changed")
    digest = model.fetch_digest(model.config.resource.model_best_weight_path)
    if digest != model.digest:
        m = load_best_model_weight(model)
        return m

    logger.debug(f"the best model is not changed")
    return False
