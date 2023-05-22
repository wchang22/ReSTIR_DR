#!/usr/bin/env python

#%%
import os
import figuregen
from figuregen import util
import simpleimageio
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

in_dir = 'positivization'
out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'font.size': 20,
    'pgf.texsystem': 'pdflatex'
})

methods=[
    'Initial',
    'Mitsuba 3',
    'Ours'
]
scenes = [
    { 'name': 'ashtray',
      'param_name': 'Anisotropy',
      'img_crop': [500, 950, 200, 200],
      'xticks': list(np.linspace(0, 30, 6)),
      'yticks': [0.010, 0.015, 0.025, 0.04, 0.07],
      'yrange': [0.009, 0.07],
      'vlim_scale': 0.01,
      'vmax_scale': 0.025,
    },
]

colors=[[255, 255, 255], [255, 0, 255], [0, 255, 255]]
crop_colors=[[225, 205, 0], [0, 225, 80], [255, 255, 255]]

def get_image(filename=None, cropbox=None, resize_square=True):
    img = simpleimageio.read(filename)
    height, width = img.shape[0], img.shape[1]
    if resize_square and width != height:
        side = min(width, height)
        img = img[(height-side)//2:height-(height-side)//2,(width-side)//2:width-(width-side)//2]
    if isinstance(cropbox, util.image.Cropbox):
        img = cropbox.crop(img)
    return img

def place_label(element: figuregen.ElementView, txt,
    pos='bottom_left', txt_color=[255,255,255], offset_mm=[0.4, 0.4],
    size_mm=[27, 6.0], fontsize=14):
    element.set_label(txt, pos, width_mm=size_mm[0], height_mm=size_mm[1], offset_mm=offset_mm,
        fontsize=fontsize, txt_color=txt_color, txt_padding_mm=0.8)

def cmap_deriv(image, vlim):
    normalize = mpl.colors.Normalize(vmin=-vlim, vmax=vlim, clip=True)
    return mpl.cm.coolwarm(normalize(image))

def cmap_diff(image, vmax_scale):
    normalize = mpl.colors.Normalize(vmin=0, vmax=vmax_scale, clip=True)
    return mpl.cm.viridis(normalize(image))

def luminance(image):
    return image[...,0] * 0.212671 + image[...,1] * 0.715160 + image[...,2] * 0.072169

for s in scenes:
    crop = util.image.Cropbox(
        left=s['img_crop'][0],
        top=s['img_crop'][1],
        width=s['img_crop'][2],
        height=s['img_crop'][3],
    )

    # Top right ----------------------------------------------------------------------------------------------
    top_right_grid = figuregen.Grid(num_rows=2, num_cols=4)
    top_right_grid.set_col_titles('top', ['Mitsuba 3', ' Ours, Without Positivization', 'Ours, With Positivization', 'Reference'])
    top_right_grid.get_layout().set_col_titles('top', field_size_mm=7, fontsize=10)

    e = top_right_grid.get_element(0, 3)
    image_gt = get_image(os.path.join(in_dir, s['name'], 'render_gt.exr'), crop)
    e.set_image(figuregen.PNG(util.image.lin_to_srgb(image_gt)))

    e = top_right_grid.get_element(1, 3)
    e.set_image(figuregen.PNG(np.ones(image_gt.shape)))

    image = get_image(os.path.join(in_dir, s['name'], 'render_final_abs.exr'), crop)
    err = luminance(util.image.relative_squared_error(image, image_gt, epsilon=1e-2))
    vmax = np.max(err) * s['vmax_scale']

    def set_inset(col, variant):
        e = top_right_grid.get_element(0, col)
        final = get_image(os.path.join(in_dir, s['name'], f'render_final_{variant}.exr'), crop)
        e.set_image(figuregen.PNG(util.image.lin_to_srgb(final)))

        e = top_right_grid.get_element(1, col)
        err = luminance(util.image.relative_squared_error(final, image_gt, epsilon=1e-2))
        e.set_image(figuregen.PNG(cmap_diff(err, vmax)))

    set_inset(0, 'mitsuba')
    set_inset(1, 'abs')
    set_inset(2, 'positivization')

    image_gt = get_image(os.path.join(in_dir, s['name'], 'render_gt.exr'))
    mitsuba_image = get_image(os.path.join(in_dir, s['name'], 'render_final_mitsuba.exr'))
    abs_image = get_image(os.path.join(in_dir, s['name'], 'render_final_abs.exr'))
    positivization_image = get_image(os.path.join(in_dir, s['name'], 'render_final_positivization.exr'))
    mitsuba_err = util.image.relative_mse(mitsuba_image, image_gt, 1e-2)
    positivization_err = util.image.relative_mse(positivization_image, image_gt, 1e-2)
    abs_err = util.image.relative_mse(abs_image, image_gt, 1e-2)

    top_right_grid.set_col_titles('bottom', [
        f'relMSE: {mitsuba_err:.1e} (1.00x)',
        f'relMSE: {abs_err:.1e} ({abs_err/mitsuba_err:.2f}x)',
        f'relMSE: {positivization_err:.1e} ({positivization_err/mitsuba_err:.2f}x)',
        '',
    ])
    top_right_grid.get_layout().set_col_titles('bottom', field_size_mm=7, fontsize=10)

    # Specify paddings (unit: mm)
    lay = top_right_grid.get_layout()
    lay.set_padding(row=0.5, column=0.5, bottom=1)

    v_grids = [
        [top_right_grid],
    ]

    figuregen.figure(v_grids, width_cm=18., filename=os.path.join(out_dir, "positivization.pdf"))

    # Colorbars --------------------------------------------------------------------------------------------------
    plt.clf()
    fig, ax = plt.figure(figsize=(3.25, 3.25), dpi=100), plt.axes()
    ax.axis('off')
    diff_image = util.image.relative_squared_error(abs_image, image_gt, 1e-2)
    err_im = ax.imshow(luminance(diff_image), cmap=mpl.cm.viridis, vmin=0, vmax=vmax)
    err_cb = fig.colorbar(err_im, ax=ax, fraction=0.045, pad=0.04, location='right')
    err_cb.set_ticks([0, vmax])
    err_cb.set_ticklabels(['0', f'{vmax:.1f}'])
    err_cb.ax.yaxis.set_offset_position('left')
    err_cb.ax.tick_params(labelsize=12)
    for tick in err_cb.ax.yaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    plt.savefig(os.path.join(out_dir, "positivization-err-cb.pdf"), bbox_inches='tight', pad_inches=0.0)

# %%