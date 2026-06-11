"""Render RL evaluation JSONL records into a standalone HTML viewer.

Usage:
    python scripts/render_eval_records.py \
        --input logs/douzero_4a4_action_embedding_eval_records.jsonl \
        --output logs/eval_viewer.html

The source JSONL is read-only. The generated HTML embeds the selected records
and uses static/rl_eval_viewer.html as its template.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
TEMPLATE = os.path.join(ROOT, 'static', 'rl_eval_viewer.html')
MARKER = '/*__EVAL_RECORDS__*/'


def _load_records(path, limit):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    '%s:%d is not valid JSON: %s' % (path, line_no, exc)
                ) from exc
    if limit and limit > 0:
        records = records[-limit:]
    return records


def _render(records, output_path):
    with open(TEMPLATE, 'r', encoding='utf-8') as f:
        html = f.read()
    payload = json.dumps(records, ensure_ascii=False)
    injection = 'window.EMBEDDED_EVAL_RECORDS = %s;' % payload
    if MARKER not in html:
        raise RuntimeError('viewer template marker not found: %s' % MARKER)
    html = html.replace(MARKER, injection)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='evaluation records JSONL path')
    parser.add_argument('--output', required=True,
                        help='standalone HTML output path')
    parser.add_argument('--limit', type=int, default=50,
                        help='only embed the latest N records; <=0 embeds all')
    args = parser.parse_args(argv)

    records = _load_records(args.input, args.limit)
    _render(records, args.output)
    print('rendered %d records to %s' % (len(records), args.output))


if __name__ == '__main__':
    main()
