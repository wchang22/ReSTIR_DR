#!/usr/bin/env python

#%%
import mitsuba as mi
import drjit as dr
import numpy as np
import os
import json

mi.set_variant('cuda_ad_rgb')

output_dir_base = 'derivatives'
os.makedirs(output_dir_base, exist_ok=True)

#%% Load scene and parameters

scenes = [
    { 'name': 'christmas-tree',
      'path': '../scenes/christmas-tree/scene.xml',
      'key': 'mat-pine.brdf_0.base_color.data',
      'restir_spp': 1,
      'mitsuba_spp': 2,
    },
    { 'name': 'tire',
      'path': '../scenes/tire/scene.xml',
      'key': 'mat-tire.brdf_0.roughness.data',
      'restir_spp': 1,
      'mitsuba_spp': 3,
      'learning_rate': 0.005,
    },
    { 'name': 'ashtray',
      'path': '../scenes/ashtray/scene.xml',
      'key': 'mat-ashtray.brdf_0.anisotropic.data',
      'restir_spp': 1,
      'mitsuba_spp': 4,
    },
    { 'name': 'chalice',
      'path': '../scenes/chalice/scene.xml',
      'key': 'mat-chalice.brdf_0.roughness.data',
      'restir_spp': 1,
      'mitsuba_spp': 3,
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

def convert_to_lum(grad_tensor, extend_dim=False):
    if len(grad_tensor.shape) != 3:
        return grad_tensor
    if grad_tensor.shape[2] == 1 and not extend_dim:
        return grad_tensor[:,:,0]
    grad_color = dr.unravel(mi.Color3f, dr.ravel(grad_tensor[...,:3]))
    grad_lum = mi.luminance(grad_color)
    shape = (grad_tensor.shape[0], grad_tensor.shape[1], 1) if extend_dim else \
        (grad_tensor.shape[0], grad_tensor.shape[1])
    return mi.TensorXf(grad_lum, shape=shape)

print(f'-------------------- Running {scene_name} -------------------------')

scene = mi.load_file(scene_path, integrator='restir_dr')

is_base_color = 'base_color' in key
if 'learning_rate' in s:
    learning_rate = s['learning_rate']
elif is_base_color:
    learning_rate = 0.1
else:
    learning_rate = 0.01

image_gt = mi.render(scene, seed=0, spp=render_spp);
mi.util.write_bitmap(os.path.join(output_dir, 'render_gt.png'), image_gt)
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
mi.util.write_bitmap(os.path.join(output_dir, 'render_initial.png'), image_initial)
mi.util.write_bitmap(os.path.join(output_dir, 'render_initial.exr'), image_initial)

opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);
scene.integrator().param_name = key
scene.integrator().materialize_grad = False

dr.set_flag(dr.JitFlag.KernelHistory, 1)

def get_elapsed_execution_time():
    hist = dr.kernel_history()
    elapsed_time = 0
    for entry in hist:
        elapsed_time += entry['execution_time']
    return elapsed_time

def relse(a, b):
    return dr.sqr(a - b) / (dr.sqr(b) + 1e-2)

def relmse(a, b):
    return dr.mean(relse(a, b))

def derivative_err(img, ref):
    return dr.sum(relse(img, ref)) / dr.count(dr.neq(ref.array, 0))

def loss_func(image):
    return relmse(image, image_gt)

def get_equal_time_derivatives(use_ref, n_iterations, spp_forward, spp_grad, M_cap=None, params_state=None):
    np.random.seed(0)
    
    # Reset initial params
    opt.reset(key)
    opt[key] = mi.TensorXf(param_initial) if params_state is None else params_state
    params.update(opt);

    scene.integrator().use_ref = use_ref
    scene.integrator().use_positivization = True
    scene.integrator().enable_temporal_reuse = True
    scene.integrator().M_cap = M_cap
    scene.integrator().reset()

    total_time = 0
    for it in range(n_iterations):
        print(f'---- Iteration {it} ---- ')

        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(scene, params, spp=spp_forward,
            spp_grad=spp_grad,
            seed=np.random.randint(2**31))

        # Evaluate the objective function from the current rendered image
        loss = loss_func(image)

        # Backpropagate through the rendering process
        dr.backward(loss)

        if it == n_iterations - 1:
            grad = dr.grad(opt[key])
            params_state = mi.TensorXf(params[key])

        # Optimizer: take a gradient descent step
        opt.step()

        # Post-process the optimized parameters to ensure legal color values.
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)

        # Update the scene state to the new optimized values
        params.update(opt)

        total_time += get_elapsed_execution_time()

    return grad, params_state, total_time / n_iterations


restir_grad, params_state, restir_execution_time = \
    get_equal_time_derivatives(False, equal_time_iterations, spp_forward, s['restir_spp'], \
        M_cap=s.get('restir_mcap', 32))

mitsuba_grad, _, mitsuba_execution_time = \
    get_equal_time_derivatives(True, 1, spp_forward, s['mitsuba_spp'], params_state=params_state)

gt_grad, _, gt_execution_time = \
    get_equal_time_derivatives(True, 1, render_spp, render_spp, params_state=params_state)

restir_err = derivative_err(restir_grad, gt_grad)[0]
mitsuba_err = derivative_err(mitsuba_grad, gt_grad)[0]
print(
    f'ReSTIR, took: {restir_execution_time:.3f} ms, error: {restir_err:.3e} ({restir_err/mitsuba_err:.2f}x)\n'
    f'Mitsuba, took: {mitsuba_execution_time:.3f} ms, error: {mitsuba_err:.3e} (1.00x)\n'
)
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    f.write(json.dumps({
        'restir_time': restir_execution_time,
        'restir_err': restir_err,
        'mitsuba_time': mitsuba_execution_time,
        'mitsuba_err': mitsuba_err,
        'err_reduction': restir_err/mitsuba_err,
    }, indent=2))


mi.util.write_bitmap(os.path.join(output_dir, 'deriv_gt.exr'), gt_grad)
mi.util.write_bitmap(os.path.join(output_dir, 'deriv_restir.exr'), restir_grad)
mi.util.write_bitmap(os.path.join(output_dir, 'deriv_mitsuba.exr'), mitsuba_grad)
