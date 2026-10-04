"""Regenerate the four editable SourceBridge figures and their preview gallery.

Run from any working directory: python source/regenerate.py
The figure generators retain the scientific examples from the current paper.
"""
from pathlib import Path
import os
import subprocess
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE


def main():
    env = os.environ.copy()
    env.setdefault('SOURCEBRIDGE_CONCEPT_OUT', str(ROOT / 'figures'))
    env.setdefault('SOURCEBRIDGE_CONCEPT_QA_OUT', str(ROOT / 'quality'))
    env.setdefault('SOURCEBRIDGE_PALETTE', 'indigo_vivid')
    env.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sourcebridge-figures-mpl'))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('system_figures.py', 'mechanism_figures.py'):
        subprocess.run([sys.executable, str(HERE / name)], env=env, check=True)
    subprocess.run([sys.executable, str(HERE / 'make_gallery.py')], env=env, check=True)
    for language in ('en','zh'):
        target=HERE.parents[1]/f'manuscript_{language}'/'figures'
        target.mkdir(parents=True,exist_ok=True)
        for source in (ROOT/'figures').iterdir():
            if source.suffix in ('.pdf','.svg','.png') and source.stem in ('motivation','workflow','acquisition','geometry'):
                shutil.copy2(source,target/source.name)
    print('Conceptual figures and manuscript figure directories regenerated.')


if __name__ == '__main__':
    main()
