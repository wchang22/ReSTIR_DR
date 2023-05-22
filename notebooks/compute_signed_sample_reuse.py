#%%
import os
import json

scenes = ['chalice', 'tire', 'ashtray', 'christmas-tree']
paths = [
    'inverse-rendering',
    'inverse-rendering-drop-incorrect-signed-samples',
]

for scene in scenes:
    print(f'----- Scene {scene} ---- ')
    errs = []
    mitsuba_err = 0
    for i, path in enumerate(paths):
        inv_convergence_path = os.path.join(path, scene, 'inv_convergence.json')

        with open(inv_convergence_path) as f:
            inv_convergence = json.loads(f.read())
            errs.append(inv_convergence['restir_img_err'])
            if i == 0:
                mitsuba_err = inv_convergence['mitsuba_img_err']

    print(f'Mitsuba 3 {mitsuba_err:.3e}')
    print(f'Ours Without {errs[1]:.2e}({errs[1]/mitsuba_err:.2f}x)')
    print(f'Ours With {errs[0]:.2e}({errs[0]/mitsuba_err:.2f}x)')
            