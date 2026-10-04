"""Verify immutable code/data hashes and parse every packaged Python source."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'code_manifest.json').read_text('utf-8'))
    for relative, expected in manifest['sha256'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    sources = list(ROOT.rglob('*.py'))
    for path in sources:
        ast.parse(path.read_text('utf-8-sig'), filename=str(path))
    print(json.dumps({'status': 'PASS', 'immutable_files': len(manifest['sha256']),
                      'python_syntax_checks': len(sources)}))


if __name__ == '__main__':
    main()
