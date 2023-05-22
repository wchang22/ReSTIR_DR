#!/usr/bin/env python

#%%
import os
import figuregen
import json
import matplotlib as mpl
import numpy as np

inv_dir ='inverse-rendering-convergence'
out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'font.size': 20,
    'pgf.texsystem': 'pdflatex'
})

colors=[[255, 255, 255], [255, 75, 255], [75, 255, 255]]
scenes = [
    { 'name': 'chalice',
      'label': '(a) Chalice',
      'xticks': list(np.linspace(0, 450, 6)[:-2]),
      'yticks': [0.08, 0.12, 0.2, 0.3, 0.4],
      'yrange': [0.07, 0.4],
    },
    { 'name': 'tire',
      'label': '(b) Tire',
      'xticks': list(np.linspace(0, 900, 6)[:-2]),
      'yticks': [0.16, 0.23, 0.33, 0.48, 0.7],
      'yrange': [0.15, 0.7],
    },
    { 'name': 'ashtray',
      'label': '(c) Ashtray',
      'xticks': list(np.linspace(0, 300, 6)[:-2]),
      'yticks': [0.012, 0.02, 0.03, 0.05, 0.07],
      'yrange': [0.011, 0.07],
    },
    { 'name': 'christmas-tree',
      'label': '(d) Christmas Tree',
      'xticks': list(np.linspace(0, 350, 6)[:-2]),
      'yticks': [0.08, 0.1, 0.2, 0.4, 0.8],
      'yrange': [0.075, 0.8],
    },
]

grid = figuregen.Grid(num_rows=2, num_cols=2)

for i, s in enumerate(scenes):
    e = grid.get_element(i//2, i%2)
    with open(os.path.join(inv_dir, s['name'], 'inv_convergence.json')) as f:
        data = json.load(f)

    plot = figuregen.PgfLinePlot(aspect_ratio=1/1.15, data=[
        (data['mitsuba_times'][::8], data['mitsuba_losses'][::8]),
        (data['restir_times'][::8], data['restir_losses'][::8])
    ])

    plot.set_linewidth(plot_line_pt=2)
    plot.set_axis_label('x', 'Time (s)')
    plot.set_axis_properties('x', ticks=s['xticks'], use_log_scale=False)
    plot.set_axis_label('y', 'Error')
    plot.set_axis_properties('y', ticks=s['yticks'], range=s['yrange'], use_log_scale=True)
    plot.set_colors(colors[1:])
    if i == 1:
        plot.set_legend(['Mitsuba 3', 'Ours'])
    plot.set_padding(left_mm=9, bottom_mm=8, right_mm=0)
    plot.set_font(8)
    e.set_label(s['label'], 'top', 30, 6, offset_mm=[-4, 1], fontsize=8)
    e.set_image(plot)

v_grids = [[grid]]

figuregen.figure(v_grids, width_cm=9., filename=os.path.join(out_dir, f"convergence.pdf"))

# %%