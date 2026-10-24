"""
Final-only postprocessor.
Does not save every load step.
Reads one final solution file and creates basic 2D/3D views.
"""

import numpy as np


def load_solution(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def summary(data):
    print('FIELDS:', list(data.keys()))
    for k, v in data.items():
        try:
            print(k, v.shape)
        except Exception:
            pass


if __name__ == '__main__':
    print('FINAL_POSTPROCESS_2D_3D')
    print('Usage: load final NPZ and generate plots from final state only.')
