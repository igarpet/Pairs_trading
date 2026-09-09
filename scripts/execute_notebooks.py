"""Execute selected root notebooks and preserve outputs, including on failure."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='+')
    args = parser.parse_args()
    for name in args.notebooks:
        path = ROOT / name
        if path.parent != ROOT or path.suffix != '.ipynb':
            raise ValueError('Expected a root notebook filename.')
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        print(f'Starting {name}', flush=True)
        try:
            NotebookClient(notebook, timeout=None, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, path)
        print(f'Completed {name}', flush=True)

if __name__ == '__main__':
    main()
