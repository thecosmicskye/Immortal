import sys

from redis import Redis
import cloudpickle as pickle
import functools

OPPONENT_MODELS = "opponent-models"

def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent

@functools.lru_cache(maxsize=8)
def _get_past_model(redis, version):
    assert isinstance(version, int)
    return _unserialize_model(redis.lindex(OPPONENT_MODELS, version))

if __name__ == '__main__':
    _, ip, password = sys.argv
    redis = Redis(host=ip, password=password)
    model = _get_past_model(redis, 898)
    print(model['epoch'])
    exit(0)
