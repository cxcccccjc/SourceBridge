"""Import scientific APIs without executing benchmark workflows."""
from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/source'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
MODULES = (
    'reference_geometry', 'catalog_selection', 'additive_selection',
    'certified_geometry', 'packet_geometry', 'safe_projection',
    'robust_doptimal', 'mlni_calibration', 'batch_baselines', 'eptd_inference',
    'group_methods', 'spptd_plaintext_core', 'air_quality_baselines',
    'acquisition_evaluation', 'weighted_inference', 'simulated_inference',
    'public_source_generation', 'public_inference', 'procurement',
    'packet_attacks', 'attack_evaluation', 'selection_scaling',
)


def main():
    for name in MODULES:
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().parent == SOURCE.resolve(), name
    print(json.dumps({'status': 'PASS', 'scientific_api_modules': len(MODULES)}))


if __name__ == '__main__':
    main()
