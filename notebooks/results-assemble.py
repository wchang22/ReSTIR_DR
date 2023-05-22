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
from skimage.transform import downscale_local_mean

derivatives_dir = 'derivatives'
inv_dir ='inverse-rendering'
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
    { 'name': 'christmas-tree',
      'param_name': 'Base Color',
      'deriv_crop1': [0, 520, 200, 200],
      'deriv_crop2': [450, 400, 200, 200],
      'img_crop': [550, 880, 300, 300],
      'xticks': list(np.linspace(0, 350, 6)),
      'yticks': [0.08, 0.1, 0.2, 0.4, 0.8],
      'yrange': [0.075, 0.8],
      'vlim_scale': 0.15,
      'vmax_scale': 0.003,
    },
    { 'name': 'tire',
      'param_name': 'Roughness',
      'deriv_crop1': [0, 1324, 700, 700],
      'deriv_crop2': [770, 200, 600, 600],
      'img_crop': [880, 475, 300, 300],
      'xticks': list(np.linspace(0, 90, 6)),
      'yticks': [0.24, 0.3, 0.4, 0.5, 0.7],
      'yrange': [0.22, 0.7],
      'vlim_scale': 0.2,
      'vmax_scale': 0.03,
    },
    { 'name': 'ashtray',
      'param_name': 'Anisotropy',
      'deriv_crop1': [100, 1250, 600, 600],
      'deriv_crop2': [1000, 925, 500, 500],
      'img_crop': [480, 800, 300, 300],
      'xticks': list(np.linspace(0, 30, 6)),
      'yticks': [0.015, 0.02, 0.03, 0.05, 0.07],
      'yrange': [0.014, 0.07],
      'vlim_scale': 0.01,
      'vmax_scale': 0.01,
    },
    { 'name': 'chalice',
      'param_name': 'Roughness',
      'deriv_crop1': [750, 400, 500, 500],
      'deriv_crop2': [1275, 300, 500, 500],
      'img_crop': [450, 150, 550, 550],
      'xticks': list(np.linspace(0, 30, 6)),
      'yticks': [0.1, 0.15, 0.2, 0.3, 0.4],
      'yrange': [0.09, 0.4],
      'vlim_scale': 1.0,
      'vmax_scale': 0.0015,
    },
]

colors=[[255, 255, 255], [255, 75, 255], [75, 255, 255]]
crop_colors=[[225, 205, 0], [0, 225, 80], [255, 255, 255]]

def get_image(filename=None, cropbox=None, resize_square=True):
    img = simpleimageio.read(filename)
    if len(img.shape) < 3:
        img = img[...,None]
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
    if image.shape[2] == 1:
        return image[:,:,0]
    return image[...,0] * 0.212671 + image[...,1] * 0.715160 + image[...,2] * 0.072169

def downscale(image):
    return downscale_local_mean(image, (2, 2, 1))

for s in scenes:
    # Top left ----------------------------------------------------------------------------------------------
    top_left_grid = figuregen.Grid(num_rows=1, num_cols=2)
    top_left_grid.set_col_titles('top', ['(a) Target', f"(b) {s['param_name']} Texture Derivatives"])

    crop = util.image.Cropbox(
        left=s['img_crop'][0]//2,
        top=s['img_crop'][1]//2,
        width=s['img_crop'][2]//2,
        height=s['img_crop'][3]//2
    )

    e = top_left_grid.get_element(0, 0)
    e.set_image(figuregen.PNG(downscale(util.image.lin_to_srgb(get_image(os.path.join(derivatives_dir, s['name'], 'render_gt.exr'))))))
    e.set_marker(crop.get_marker_pos(), crop.get_marker_size(), color=crop_colors[-1], linewidth_pt=0.8)

    e = top_left_grid.get_element(0, 1)
    deriv_ref = luminance(get_image(os.path.join(derivatives_dir, s['name'], 'deriv_gt.exr')))
    vlim = np.max(np.abs(deriv_ref)) * 1e-3 * s['vlim_scale']

    deriv_ref_cmapped = cmap_deriv(deriv_ref, vlim)
    deriv_ref_cmapped[:,:,3]=1
    e.set_image(figuregen.PNG(downscale(deriv_ref_cmapped)))

    for crop_i in range(2):
        crop = util.image.Cropbox(
            left=s[f'deriv_crop{crop_i+1}'][0]//2,
            top=s[f'deriv_crop{crop_i+1}'][1]//2,
            width=s[f'deriv_crop{crop_i+1}'][2]//2,
            height=s[f'deriv_crop{crop_i+1}'][3]//2
        )
        e.set_marker(crop.get_marker_pos(), crop.get_marker_size(), color=crop_colors[crop_i], linewidth_pt=0.8)

    # Specify paddings (unit: mm)
    lay = top_left_grid.get_layout()
    lay.set_padding(right=1, bottom=2, column=12)

    # Top right ----------------------------------------------------------------------------------------------
    top_right_grid = figuregen.Grid(num_rows=2, num_cols=3)
    top_right_grid.set_col_titles('top', ['Mitsuba 3', 'Ours', 'Reference'])
    with open(os.path.join(derivatives_dir, s['name'], 'results.json')) as f:
        results = json.loads(f.read())
        restir_err, mitsuba_err = results['restir_err'], results['mitsuba_err']
        top_right_grid.set_col_titles('bottom', [
            f'relMSE: {mitsuba_err:.1e} (1.00x)',
            f'relMSE: {restir_err:.1e} ({restir_err/mitsuba_err:.2f}x)',
            ''
        ])

    # fill grid with image data
    for row in range(2):
        crop = util.image.Cropbox(
            left=s[f'deriv_crop{row+1}'][0],
            top=s[f'deriv_crop{row+1}'][1],
            width=s[f'deriv_crop{row+1}'][2],
            height=s[f'deriv_crop{row+1}'][3],
        )

        def set_inset(col, variant):
            e = top_right_grid.get_element(row, col)
            deriv = get_image(os.path.join(derivatives_dir, s['name'], f'deriv_{variant}.exr'), crop)
            deriv = cmap_deriv(luminance(deriv), vlim)
            e.set_image(figuregen.PNG(deriv))
            e.set_frame(linewidth=0.8, color=crop_colors[row])

        set_inset(0, 'mitsuba')
        set_inset(1, 'restir')
        set_inset(2, 'gt')

    # Specify paddings (unit: mm)
    lay = top_right_grid.get_layout()
    lay.set_padding(row=0.5, column=0.5, bottom=2)

    # Bottom left ----------------------------------------------------------------------------------------------
    bottom_left_grid = figuregen.Grid(num_rows=2, num_cols=4)
    bottom_left_grid.get_layout().set_title('top', field_size_mm=3, fontsize=7)
    bottom_left_grid.set_title('top', '(c) Reconstructed Images')
    bottom_left_grid.set_col_titles('top', ['Initial', 'Mitsuba 3', 'Ours', 'Reference'])

    crop = util.image.Cropbox(
        left=s['img_crop'][0],
        top=s['img_crop'][1],
        width=s['img_crop'][2],
        height=s['img_crop'][3]
    )

    e = bottom_left_grid.get_element(0, 3)
    image_gt = get_image(os.path.join(derivatives_dir, s['name'], 'render_gt.exr'), crop)
    e.set_image(figuregen.PNG(util.image.lin_to_srgb(image_gt)))

    e = bottom_left_grid.get_element(1, 3)
    e.set_image(figuregen.PNG(np.ones(image_gt.shape)))

    e = bottom_left_grid.get_element(0, 0)
    image_initial = get_image(os.path.join(derivatives_dir, s['name'], 'render_initial.exr'), crop)
    e.set_image(figuregen.PNG(util.image.lin_to_srgb(image_initial)))

    e = bottom_left_grid.get_element(1, 0)
    err = luminance(util.image.relative_squared_error(image_initial, image_gt, epsilon=1e-2))
    vmax = np.max(err) * s['vmax_scale']
    e.set_image(figuregen.PNG(cmap_diff(err, vmax)))

    def set_inset(col, variant):
        e = bottom_left_grid.get_element(0, col)
        final = get_image(os.path.join(inv_dir, s['name'], f'render_final_{variant}.exr'), crop)
        e.set_image(figuregen.PNG(util.image.lin_to_srgb(final)))

        e = bottom_left_grid.get_element(1, col)
        err = luminance(util.image.relative_squared_error(final, image_gt, epsilon=1e-2))
        e.set_image(figuregen.PNG(cmap_diff(err, vmax)))

    mitsuba_err = set_inset(1, 'mitsuba')
    restir_err = set_inset(2, 'restir')

    image_gt = get_image(os.path.join(derivatives_dir, s['name'], 'render_gt.exr'))
    mitsuba_image = get_image(os.path.join(inv_dir, s['name'], 'render_final_mitsuba.exr'))
    restir_image = get_image(os.path.join(inv_dir, s['name'], 'render_final_restir.exr'))
    restir_err = util.image.relative_mse(restir_image, image_gt, 1e-2)
    mitsuba_err = util.image.relative_mse(mitsuba_image, image_gt, 1e-2)

    bottom_left_grid.set_col_titles('bottom', [
        '',
        f'relMSE: {mitsuba_err:.1e} (1.00x)',
        f'relMSE: {restir_err:.1e} ({restir_err/mitsuba_err:.2f}x)',
        '',
    ])

    # Specify paddings (unit: mm)
    lay = bottom_left_grid.get_layout()
    lay.set_padding(row=0.5, column=0.5)

    # Bottom right ----------------------------------------------------------------------------------------------
    bottom_right_grid = figuregen.Grid(num_rows=1, num_cols=1)
    bottom_right_grid.set_col_titles('top', ['(d) Inverse Rendering Convergence'])

    e = bottom_right_grid.get_element(0, 0)
    with open(os.path.join(inv_dir, s['name'], 'inv_convergence.json')) as f:
        data = json.load(f)
    plot = figuregen.PgfLinePlot(aspect_ratio=1/1.15, data=[
        (data['mitsuba_times'][::4], data['mitsuba_losses'][::4]),
        (data['restir_times'][::4], data['restir_losses'][::4])
    ])
    plot.set_linewidth(plot_line_pt=2)
    plot.set_axis_label('x', 'Time (s)')
    plot.set_axis_properties('x', ticks=s['xticks'], use_log_scale=False)
    plot.set_axis_label('y', 'Error')
    plot.set_axis_properties('y', ticks=s['yticks'], range=s['yrange'], use_log_scale=True)
    plot.set_colors(colors[1:])
    plot.set_legend(['Mitsuba 3', 'Ours'])
    plot.set_padding(left_mm=9, bottom_mm=4, right_mm=0)
    e.set_image(plot)

    v_grids = [
        [top_left_grid, top_right_grid],
        [bottom_left_grid, bottom_right_grid]
    ]

    figuregen.figure(v_grids, width_cm=18., filename=os.path.join(out_dir, f"{s['name']}.pdf"))

    # Colorbars --------------------------------------------------------------------------------------------------
    plt.clf()
    fig, ax = plt.figure(figsize=(2.75, 2.75), dpi=100), plt.axes()
    ax.axis('off')
    deriv_im = ax.imshow(deriv_ref, cmap=mpl.cm.coolwarm, vmin=-vlim, vmax=vlim)
    deriv_cb = fig.colorbar(deriv_im, ax=ax, fraction=0.045, pad=0.04, location='left')
    deriv_cb.ax.locator_params(nbins=3)
    deriv_cb.ax.yaxis.set_offset_position('right')
    deriv_cb.ax.tick_params(labelsize=7)
    deriv_cb.ax.yaxis.offsetText.set_fontsize(7)
    plt.savefig(os.path.join(out_dir, f"{s['name']}-deriv-cb.pdf"), bbox_inches='tight', pad_inches=0.0)

    plt.clf()
    fig, ax = plt.figure(figsize=(1.4, 1.4), dpi=100), plt.axes()
    ax.axis('off')
    diff_image = util.image.relative_squared_error(restir_image, image_gt, 1e-2)
    err_im = ax.imshow(luminance(diff_image), cmap=mpl.cm.viridis, vmin=0, vmax=vmax)
    err_cb = fig.colorbar(err_im, ax=ax, fraction=0.045, pad=0.04, location='right')
    err_cb.set_ticks([0, vmax])
    err_cb.set_ticklabels(['0', f'{vmax:.1f}'])
    err_cb.ax.yaxis.set_offset_position('left')
    err_cb.ax.tick_params(labelsize=4)
    for tick in err_cb.ax.yaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    plt.savefig(os.path.join(out_dir, f"{s['name']}-err-cb.pdf"), bbox_inches='tight', pad_inches=0.0)

# %%