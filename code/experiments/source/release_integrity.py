"""Resolve immutable resource identifiers and verify distributed source identities.

Frozen protocols retain their original hashes. Source files have separate hashes
because descriptive names, imports, and portable paths have been standardized.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'source':
    ROOT = ROOT.parent


def resolve_resource(path):
    """Resolve an archived basename without changing a frozen record."""
    path = Path(path)
    if path.exists():
        return path
    mapping = json.loads((ROOT / 'resource_names.json').read_text('utf-8'))
    return path.with_name(mapping.get(path.name, path.name))


def verify_identity(path, expected):
    """Check bytes against the release hash and provenance against the protocol."""
    path = resolve_resource(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        return True
    records = json.loads((ROOT / 'source_identity.json').read_text('utf-8'))
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    record = records.get(relative)
    return bool(record and expected == record['original_sha256']
                and actual == record['release_sha256'])
