"""Print high-level metadata from a DouZero-4A4 checkpoint."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.model import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    if torch is None:
        raise RuntimeError('torch is required to inspect checkpoints')
    payload = torch.load(args.checkpoint, map_location='cpu')
    state_keys = sorted(k for k in payload.keys() if k.endswith('state_dict'))
    print('state_dict_keys=%s' % ','.join(state_keys))
    print('model_type=%s' % payload.get('model_type'))
    print('backend_model_class=%s' % payload.get('backend_model_class'))
    print('teacher_model_class=%s' % payload.get('teacher_model_class'))
    print('global_step=%s' % payload.get('global_step'))
    print('episode=%s' % payload.get('episode'))


if __name__ == '__main__':
    main()

