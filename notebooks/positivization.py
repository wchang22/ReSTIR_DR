#!/usr/bin/env python

#%%
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import os

mi.set_variant('cuda_ad_rgb')

output_dir_base = 'positivization'
os.makedirs(output_dir_base, exist_ok=True)

#%% Load scene and parameters

scenes = [
    { 'name': 'ashtray',
      'path': '../scenes/ashtray/scene.xml',
      'key': 'mat-ashtray.brdf_0.anisotropic.data',
      'restir_spp': 1,
      'mitsuba_spp': 1,
      'time': 50e3,
    },
]
scene_idx = 0

render_spp = 512
spp_forward = 32
equal_time_iterations = 20

s = scenes[scene_idx]
scene_path, scene_name, key = s['path'], s['name'], s['key']
output_dir = os.path.join(output_dir_base, scene_name)
os.makedirs(output_dir, exist_ok=True)

print(f'-------------------- Running {scene_name} -------------------------')

scene = mi.load_file(scene_path, integrator='restir_dr')

learning_rate = 0.01

image_gt = mi.render(scene, seed=0, spp=render_spp);
mi.util.write_bitmap(os.path.join(output_dir, 'render_gt.exr'), image_gt)

params = mi.traverse(scene)
param_ref = mi.TensorXf(params[key])
param_shape = np.array(params[key].shape)

param_initial = np.full(param_shape.tolist(), 0.5)
if param_shape[2] == 4:
    param_initial[:,:,3] = 1
    param_ref[:,:,3] = 1
params[key] = mi.TensorXf(param_initial)

params.update();

image_initial = mi.render(scene, seed=0, spp=render_spp);

opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);
scene.integrator().param_name = key

dr.set_flag(dr.JitFlag.KernelHistory, 1)

def get_elapsed_execution_time():
    hist = dr.kernel_history()
    elapsed_time = 0
    for entry in hist:
        elapsed_time += entry['execution_time']
    return elapsed_time

def convert_to_lum(grad_tensor):
    if len(grad_tensor.shape) != 3 or grad_tensor.shape[2] == 1:
        return grad_tensor
    grad_color = dr.unravel(mi.Color3f, dr.ravel(grad_tensor[...,:3]))
    grad_lum = mi.luminance(grad_color)
    return mi.TensorXf(grad_lum, shape=(grad_tensor.shape[0], grad_tensor.shape[1]))

def relse(a, b):
    return dr.sqr(a - b) / (dr.sqr(b) + 1e-2)

def relmse(a, b):
    return dr.mean(relse(a, b))

def derivative_err(img, ref):
    return dr.sum(relse(img, ref)) / dr.count(dr.neq(ref.array, 0))

def loss_func(image):
    return relmse(image, image_gt)

params[key] = mi.TensorXf(param_initial)
params.update();

opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);

def get_equal_time_optimization(use_ref, use_positivization, n_time, spp_forward, spp_grad, M_cap=None):
    np.random.seed(0)
    
    # Reset initial params
    opt.reset(key)
    opt[key] = mi.TensorXf(param_initial)
    params.update(opt);

    scene.integrator().use_ref = use_ref
    scene.integrator().use_positivization = use_positivization
    scene.integrator().enable_temporal_reuse = True
    scene.integrator().M_cap = M_cap
    scene.integrator().reset()

    it = 0
    total_time = 0
    times = []
    losses = []
    param_errs = []
    while True:
        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(scene, params, spp=spp_forward,
            spp_grad=spp_grad,
            seed=np.random.randint(2**31))

        # Evaluate the objective function from the current rendered image
        loss = loss_func(image)

        # Backpropagate through the rendering process
        dr.backward(loss)

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        # Update the scene state to the new optimized values
        params.update(opt)

        total_time += get_elapsed_execution_time()
        if total_time > n_time:
            break
        times.append(total_time / 1e3)
        losses.append(loss[0])
        param_errs.append(relmse(params[key], param_ref)[0])

        print(f'-- Iteration {it} -- Loss {losses[-1]:.3f} --')

        it += 1

    return times, losses, param_errs, mi.TensorXf(params[key])

#%% Run equal time optimization

positivization_times, positivization_losses, positivization_param_errs, positivization_param = \
    get_equal_time_optimization(False, True, s['time'], spp_forward, s['restir_spp'], M_cap=s.get('restir_mcap', 16))

abs_times, abs_losses, abs_param_errs, abs_param = \
    get_equal_time_optimization(False, False, s['time'], spp_forward, s['restir_spp'], M_cap=s.get('restir_mcap', 16))

mitsuba_times, mitsuba_losses, mitsuba_param_errs, mitsuba_param = \
    get_equal_time_optimization(True, False, s['time'], spp_forward, s['mitsuba_spp']) 

#%% Output equal time optimization
plt.clf()
plt.figure(figsize=(10, 4), dpi=100, constrained_layout=True);
plt.plot(abs_times, abs_losses, 'm-o', label='Without Positivization', linewidth=6.0, markersize=4.0, mfc='white')
plt.plot(positivization_times, positivization_losses, 'c-o', label='With Positivization', linewidth=6.0, markersize=4.0, mfc='white')
plt.plot(mitsuba_times, mitsuba_losses, 'r-o', label='Mitsuba 3', linewidth=6.0, markersize=4.0, mfc='white')
plt.xlabel('Time (s)');
plt.ylabel('Error');
plt.yscale('log')
plt.legend();
plt.savefig(os.path.join(output_dir, 'inv_convergence.pdf'), bbox_inches='tight', pad_inches=0.0)

#%% Output parameter_error
plt.clf()
plt.figure(figsize=(10, 4), dpi=100, constrained_layout=True);
plt.plot(abs_times, abs_param_errs, 'm-o', label='Without Positivization', linewidth=6.0, markersize=4.0, mfc='white')
plt.plot(positivization_times, positivization_param_errs, 'c-o', label='With Positivization', linewidth=6.0, markersize=4.0, mfc='white')
plt.plot(mitsuba_times, mitsuba_param_errs, 'r-o', label='Mitsuba 3', linewidth=6.0, markersize=4.0, mfc='white')
plt.xlabel('Time (s)');
plt.ylabel('Error');
plt.yscale('log')
plt.legend();
plt.savefig(os.path.join(output_dir, 'param_convergence.pdf'), bbox_inches='tight', pad_inches=0.0)

#%% Output equal time final image
params[key] = abs_param
params.update();
abs_image = mi.render(scene, seed=0, spp=render_spp);
mi.util.write_bitmap(os.path.join(output_dir, 'render_final_abs.exr'), abs_image)

params[key] = positivization_param
params.update();
positivization_image = mi.render(scene, seed=0, spp=render_spp);
mi.util.write_bitmap(os.path.join(output_dir, 'render_final_positivization.exr'), positivization_image)

params[key] = mitsuba_param
params.update();
mitsuba_image = mi.render(scene, seed=0, spp=render_spp);
mi.util.write_bitmap(os.path.join(output_dir, 'render_final_mitsuba.exr'), mitsuba_image)

#%%
mitsuba_img_err = loss_func(mitsuba_image)[0]
positivization_img_err = loss_func(positivization_image)[0]
abs_img_err = loss_func(abs_image)[0]

print(
    f'Mitsuba, error: {mitsuba_img_err:.3e} (1.00x)\n'
    f'With, error: {positivization_img_err:.3e} ({positivization_img_err/mitsuba_img_err:.2f}x)\n'
    f'Without, error: {abs_img_err:.3e} ({abs_img_err/mitsuba_img_err:.2f}x)\n'
)
