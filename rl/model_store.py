"""
模型加载与共享缓存。

关键要求：无论在线多少个AI玩家，同一路径/同一名称的模型在内存中只加载一份。
这里使用进程级字典 + 锁实现单例缓存。
"""
import os
import threading
from rl.model import CardPolicyNetwork, torch

DEFAULT_MODEL_NAME = 'default_rl'
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'checkpoints', 'default_rl.pt')

_model_cache = {}
_cache_lock = threading.Lock()


def resolve_model_path(model_name=None):
    """根据模型名称或checkpoint相对路径解析权重路径"""
    if not model_name or model_name in ('rule', 'heuristic'):
        return None
    root = os.path.dirname(os.path.dirname(__file__))
    checkpoints_dir = os.path.join(root, 'checkpoints')
    if os.path.isabs(model_name):
        return model_name
    if model_name == DEFAULT_MODEL_NAME:
        return DEFAULT_MODEL_PATH
    if model_name.endswith('.pt') or model_name.endswith('.pth'):
        safe_name = os.path.basename(model_name)
        return os.path.join(checkpoints_dir, safe_name)
    return os.path.join(checkpoints_dir, model_name + '.pt')


def get_shared_model(model_name=DEFAULT_MODEL_NAME):
    """获取共享模型实例；同一路径只会被加载一次"""
    if torch is None:
        raise ImportError('深度学习AI需要安装 torch，请先执行 pip install -r requirements.txt')

    model_path = resolve_model_path(model_name)
    if model_path is None:
        return None

    cache_key = os.path.abspath(model_path)
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

        model = CardPolicyNetwork()
        if os.path.exists(model_path):
            payload = torch.load(model_path, map_location='cpu')
            state_dict = payload.get('model_state_dict', payload)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # checkpoint结构与当前模型不兼容时，只加载形状匹配的参数
                current = model.state_dict()
                compatible = {
                    k: v for k, v in state_dict.items()
                    if k in current and current[k].shape == v.shape
                }
                current.update(compatible)
                model.load_state_dict(current)
        model.eval()
        _model_cache[cache_key] = model
        return model


def clear_model_cache():
    """训练或调试时可手动清空缓存"""
    with _cache_lock:
        _model_cache.clear()


def cached_model_count():
    """返回当前进程内已加载模型数量，便于验证单例加载"""
    with _cache_lock:
        return len(_model_cache)
