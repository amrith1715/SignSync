"""Repeatable evaluation; filenames supply ground truth only, never predictions."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from recognition_service import RecognitionRuntime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/static_signs_results.json')
    parser.add_argument('--url', help='Optional local server origin, e.g. http://127.0.0.1:8001')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = sorted((root / 'static_signs_images').rglob('*'))
    runtime = None if args.url else RecognitionRuntime()
    rows = []
    try:
        for path in files:
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            expected = path.stem.upper()
            row = dict(file=str(path.relative_to(root)), expected=expected,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            try:
                # No expected label or filename is passed to inference.
                if args.url:
                    request = urllib.request.Request(args.url.rstrip('/') + '/api/test-image',
                        data=path.read_bytes(), headers={'Content-Type': 'application/octet-stream'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        result = json.load(response)
                else:
                    result = runtime.test_image(path.read_bytes())
                row.update(predicted=result['predicted'], confidence=result['confidence'])
                row['correct'] = result['predicted'] == expected
                row['reason'] = ('Correct' if row['correct'] else
                    'MLP confidence below acceptance threshold' if not result['predicted'] else
                    'MLP classified extracted landmarks as a different letter; exact cause requires further evidence')
            except Exception as exc:
                row.update(predicted=None, confidence=None, correct=False, reason=str(exc))
            rows.append(row)
            print(expected, row['predicted'], row['confidence'], flush=True)
    finally:
        if runtime:
            runtime.close()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['Static signs automated test',
             f'Pipeline: {args.url or "local RecognitionRuntime.test_image"}',
             f'Correct: {sum(r["correct"] for r in rows)}/{len(rows)}',
             'Expected labels are read from filenames only for evaluation, never passed to inference.', '']
    lines += [f'{r["expected"]}: predicted {r["predicted"] or "NONE"}; confidence {r["confidence"]}; {r["reason"]}' for r in rows]
    output.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Correct: {sum(r["correct"] for r in rows)}/{len(rows)}')

if __name__ == '__main__':
    main()
