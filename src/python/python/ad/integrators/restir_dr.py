from __future__ import annotations
from typing import Tuple, Union, Optional, Any  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import gc

from .common import RBIntegrator, mis_weight


class SampleTech:
    LIGHT = 0
    BSDF = 1


class ReStirDRIntegrator(RBIntegrator):
    r"""
    .. _integrator-restir_dr:

    ReSTIR Differentiable Rendering (:monosp:`restir_dr`)
    ---------------------------------------------------------
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        self.use_ref = props.get('use_ref', False)
        self.param_name = props.get('param_name')
        self.M_cap = props.get('M_cap', 20)
        self.use_positivization = props.get('use_positivization', True)
        self.enable_temporal_reuse = props.get('enable_temporal_reuse', True)

        self.drop_incorrect_signed_samples = props.get(
            'drop_incorrect_signed_samples', False)
        self.materialize_grad = props.get('materialize_grad', True)

        self.reset()

    def to_string(self):
        return f'{type(self).__name__}[use_ref = { self.use_ref },' \
               f' param_name = { self.param_name },' \
               f' M_cap = { self.M_cap },' \
               f' use_positivization = { self.use_positivization },' \
               f' enable_temporal_reuse = { self.enable_temporal_reuse }]'

    def reset(self):
        # Intermediate and history state
        self.restir_initialized = False

        self.temporal_reservoir: mi.Reservoir = None
        self.candidate_reservoir: mi.Reservoir = None

        self.param_tensor_grad: mi.TensorXf = None
        self.temporal_param_tensor: mi.TensorXf = None
        self.temporal_grad_in: mi.Spectrum = None
        self.temporal_pos: mi.Point2f = None

        self.n_params = 0
        self.reuse_texture = None

    def eval_sample(self,
                    primal: mi.Bool,
                    scene: mi.Scene,
                    si: mi.SurfaceInteraction3f,
                    uv_em: mi.Point2f,
                    uv_bsdf: mi.Point2f,
                    w_bsdf: mi.Float,
                    active_em: mi.Bool,
                    active_bsdf: mi.Bool):
        """
        Evaluates a direct lighting sample (light and/or bsdf) specified by
        shading point si, and random numbers uvw,
        returning the MIS weighted contribution, as well as texel indices, and random numbers
        """
        bsdf_ctx = mi.BSDFContext()

        # ---------------------- Light Sampling ----------------------
        bsdf: mi.BSDF = si.bsdf()
        active_em = active_em & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        ds_emitter: mi.DirectionSample3f
        ds_emitter, emitter_value = scene.sample_emitter_direction(
            si, uv_em, True, active_em)
        with dr.resume_grad(when=not primal):
            active_em &= dr.neq(ds_emitter.pdf, 0)

            # Re-compute attached `weight_em` to enable emitter optimization
            if not primal:
                ds_emitter.d = dr.normalize(ds_emitter.p - si.p)
                val_em = scene.eval_emitter_direction(
                    si, ds_emitter, active_em)
                emitter_value = dr.select(
                    dr.neq(ds_emitter.pdf, 0), val_em / ds_emitter.pdf, 0)
                dr.disable_grad(ds_emitter.d)

            wo = si.to_local(ds_emitter.d)
            bsdf_value, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis = dr.select(ds_emitter.delta, 1.0,
                            mis_weight(ds_emitter.pdf, bsdf_pdf))
            Ld_emitter = mis * emitter_value * bsdf_value

        emitter_texel_index = None
        uvw_em = None
        if not primal and self.reuse_texture is not None:
            emitter_texel_index = bsdf.get_texel_index(
                si, self.reuse_texture, active_em)
            uvw_em = mi.Point3f(uv_em.x, uv_em.y, 0)

        # ---------------------- BSDF Sampling ----------------------
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, w_bsdf, uv_bsdf, active_bsdf)
        ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))

        with dr.resume_grad(when=not primal):
            active_bsdf &= dr.any(dr.neq(bsdf_weight, 0.0))

            # Re-compute attached `bsdf_weight`
            if not primal:
                wo = si.to_local(ray_bsdf.d)
                bsdf_val, bsdf_pdf = bsdf.eval_pdf(
                    bsdf_ctx, si, wo, active_bsdf)
                bsdf_weight = dr.select(
                    dr.neq(bsdf_pdf, 0.0), bsdf_val / dr.detach(bsdf_pdf), 0.0)

            si_bsdf: mi.SurfaceInteraction3f = scene.ray_intersect(
                ray_bsdf, active_bsdf)
            L_bsdf = si_bsdf.emitter(scene).eval(si_bsdf, active_bsdf)

            with dr.suspend_grad():
                ds_bsdf = mi.DirectionSample3f(scene, si_bsdf, si)
                delta = mi.has_flag(
                    bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
                emitter_pdf = scene.pdf_emitter_direction(
                    si, ds_bsdf, active_bsdf & ~delta)
                mis = mis_weight(bsdf_sample.pdf, emitter_pdf)

            Ld_bsdf = mis * L_bsdf * bsdf_weight

        bsdf_texel_index = None
        uvw_bsdf = None
        if not primal and self.reuse_texture is not None:
            bsdf_texel_index = bsdf.get_texel_index(
                si, self.reuse_texture, active_bsdf)
            uvw_bsdf = mi.Point3f(uv_bsdf.x, uv_bsdf.y, w_bsdf)

        return Ld_emitter, Ld_bsdf, \
            (emitter_texel_index, bsdf_texel_index), \
            (uvw_em, uvw_bsdf), \
            active_em, active_bsdf

    def ris_target_func(self, sample_value):
        """
        Computes the scalar target function value given the vector-valued sample value
        """
        return mi.luminance(sample_value)

    def get_reservoir_length(self):
        """
        Computes the total size of the reservoir buffer. This is simply the number of parameters
        when not applying positivization, and 2x the number of parameters when using positivization.
        """
        assert self.n_params != 0
        if self.use_positivization:
            return self.n_params * 2
        return self.n_params

    def eval_sample_grad(self,
                         scene: mi.Scene,
                         param_tensor: mi.TensorXf,
                         w: mi.Spectrum,
                         si: mi.SurfaceInteraction3f,
                         reservoir: mi.Reservoir):
        """
        Evaluates the derivative of the sample stored in the reservoir,
        i.e. eqn 5 in the paper
        """
        is_light_sample, valid = reservoir.is_light_sample, reservoir.valid

        rand_uv = mi.Point2f(reservoir.uvw.x, reservoir.uvw.y)
        rand_w = reservoir.uvw.z

        # Random replay shift - reusing random numbers
        Ld_emitter, Ld_bsdf, _, _, _, _ = \
            self.eval_sample(False, scene,
                             si, rand_uv, rand_uv, rand_w,
                             valid & is_light_sample,
                             valid & ~is_light_sample)

        with dr.resume_grad():
            f = dr.select(is_light_sample, Ld_emitter, Ld_bsdf)

            dr.forward(param_tensor)
            grad = w * dr.grad(f)

            if self.materialize_grad:
                # Materialize the gradient - if we don't do this,
                # this causes kernel recompilation every time
                # But doing this is quite slow
                dr.eval(grad)
            return grad

    def sample(self,
               scene: mi.Scene,
               params: Optional[mi.SceneParameters],
               mode: dr.ADMode,
               sampler: mi.Sampler,
               si: mi.SurfaceInteraction3f,
               w: Optional[mi.Spectrum],
               sample_idx: Optional[mi.UInt32],
               **kwargs  # Absorbs unused arguments
               ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        """,
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # ---------------------- Direct emission ----------------------
        with dr.resume_grad(when=not primal):
            Le = si.emitter(scene).eval(si)

        # ---------------------- Direct lighting sampling  ----------------------
        active_next = si.is_valid()

        rand_uv_em = sampler.next_2d(active_next)
        rand_uv_bsdf = sampler.next_2d(active_next)
        rand_w_bsdf = sampler.next_1d(active_next)

        Ld_emitter, Ld_bsdf, texel_indices, uvws, active_em, active_bsdf = \
            self.eval_sample(primal, scene,
                             si, rand_uv_em, rand_uv_bsdf, rand_w_bsdf,
                             active_next, active_next)

        # ------------------ Differential phase only ------------------
        if not primal:
            spp = sampler.sample_count()
            with dr.resume_grad():
                if self.use_ref:
                    L = Le + Ld_emitter + Ld_bsdf
                    dr.backward(w * L / spp)

                else:  # Perform PRIS: Alg 2, Eqn 11, 14 in the paper
                    param_tensor: mi.TensorXf = params.get(self.param_name)

                    # Use backward mode for direct emission - no resampling needed
                    if dr.grad_enabled(Le):
                        dr.backward(w * Le / spp)
                        self.param_tensor_grad = dr.grad(param_tensor)
                    else:
                        self.param_tensor_grad = mi.TensorXf(
                            0, param_tensor.shape)

                    # Forward mode for direct lighting
                    dr.forward(param_tensor)

                    # Create reservoir
                    self.candidate_reservoir = mi.Reservoir(
                        self.get_reservoir_length())

                    def update_grad_with_sample(tech, f, uvw, texel_index, active):
                        grad = w * dr.grad(f)

                        # The target value comes premultiplied by the inverse candidate pdf, q / p
                        # This allows implicit computation of the random replay Jacobian when
                        # applying reuse
                        target_value = self.ris_target_func(grad)

                        if self.use_positivization:
                            # For positivization, we store the positive samples in the lower half
                            # and the negative samples in the upper half
                            texel_index[target_value < 0] += self.n_params

                        random_num = sampler.next_1d()

                        dr.eval(uvw, random_num, texel_index,
                                active, target_value, grad)
                        self.candidate_reservoir.update(
                            tech == SampleTech.LIGHT, dr.abs(target_value),
                            sample_idx, grad, uvw, random_num, texel_index, active)

                    # Emitter sampling
                    update_grad_with_sample(
                        SampleTech.LIGHT, Ld_emitter, uvws[0], texel_indices[0], active_em)

                    # BSDF sampling
                    update_grad_with_sample(
                        SampleTech.BSDF, Ld_bsdf, uvws[1], texel_indices[1], active_bsdf)

                    # This should technically be spp * (number of pixels), but the number of pixels
                    # cancels out in the sensor response, and so simply set this to spp
                    self.candidate_reservoir.M = spp

        L = Le + Ld_emitter + Ld_bsdf

        return (
            L if primal else w,
            active_next,
            L
        )

    def do_temporal_reuse(self,
                          scene: mi.Scene,
                          sensor: mi.Sensor,
                          sampler: mi.Sampler,
                          params: mi.SceneParameters,
                          cur_grad_in: mi.TensorXf,
                          cur_si: mi.SurfaceInteraction3f,
                          cur_pos: mi.Point2f):
        """
        Applies iteration (temporal) reuse, Alg 3 in the paper
        """
        param_tensor: mi.TensorXf = params.get(self.param_name)

        # Compute sample and target values of sample at a given neighbor
        def compute_values_at_neighbor(is_neighbor: bool, reservoir: mi.Reservoir,
                                       grad_in: mi.Spectrum, pos: mi.Point2f,
                                       si: mi.SurfaceInteraction3f, pt: mi.TensorXf):
            w = self.compute_w(pos, grad_in, sensor, sampler.sample_count(),
                               weight_divide=False, active=reservoir.valid)
            if is_neighbor:
                params[self.param_name] = pt
            sample_value = self.eval_sample_grad(scene, pt, w, si, reservoir)
            if is_neighbor:
                params[self.param_name] = param_tensor
            return sample_value, self.ris_target_func(sample_value)

        def compute_mis_weights(candidate_M, candidate_target_value_at_cur,
                                candidate_target_value_at_neighbor,
                                temporal_M, temporal_target_value_at_cur,
                                temporal_target_value_at_neighbor):

            def balance_heuristic(n1, p1, n2, p2):
                np1 = n1 * p1
                return dr.select(dr.eq(np1, 0), 0, np1 / (np1 + n2 * p2))

            # Apply balance heuristic on candidates using the unnormalized target function in
            # place of pdfs
            # Jacobian determinant used for the inverse shift satisfied by the
            # premultiplied candidate pdfs in the target values
            temporal_mis_weight = balance_heuristic(temporal_M, temporal_target_value_at_neighbor,
                                                    candidate_M, temporal_target_value_at_cur)
            candidate_mis_weight = balance_heuristic(candidate_M, candidate_target_value_at_cur,
                                                     temporal_M, candidate_target_value_at_neighbor)

            return candidate_mis_weight, temporal_mis_weight

        temporal_reservoir = self.temporal_reservoir

        # Temporal history weight clamp
        temporal_reservoir.M = dr.minimum(temporal_reservoir.M, self.M_cap)

        # ----------------- Evaluate temporal sample value at current iteration -------------------

        # Reuse previous pixel location and surface interaction
        # We recompute si so that it does not need to be stored
        temporal_si_ray, _ = self.sample_ray_from_screen_pos(
            sensor, sampler, self.temporal_pos)
        temporal_si = scene.ray_intersect(
            temporal_si_ray, temporal_reservoir.valid)
        temporal_sample_value_at_cur, temporal_target_value_at_cur = compute_values_at_neighbor(
            False, temporal_reservoir, cur_grad_in, self.temporal_pos, temporal_si, param_tensor)

        # ------------------------------- Compute MIS weights -------------------------------------
        candidate_target_value_at_cur = self.ris_target_func(
            self.candidate_reservoir.sample_value)
        _, candidate_target_value_at_neighbor = compute_values_at_neighbor(
            True, self.candidate_reservoir,
            self.temporal_grad_in, cur_pos,
            cur_si, self.temporal_param_tensor)

        candidate_mis_weight, temporal_mis_weight = compute_mis_weights(
            self.candidate_reservoir.M, dr.abs(candidate_target_value_at_cur),
            dr.abs(candidate_target_value_at_neighbor),
            temporal_reservoir.M, dr.abs(temporal_target_value_at_cur),
            dr.abs(self.ris_target_func(temporal_reservoir.sample_value)))

        # -------------------- Apply MIS weights to candidate reservoirs --------------------------
        self.candidate_reservoir.weight_sum *= candidate_mis_weight / \
            self.candidate_reservoir.M

        reservoir = self.candidate_reservoir

        # ------------------ Apply MIS weights and merge previous reservoirs ----------------------
        index = dr.arange(mi.UInt32, self.get_reservoir_length())
        if self.use_positivization:
            if self.drop_incorrect_signed_samples:  # Simply zero out samples with the incorrect sign
                sign = dr.select(index < self.n_params, 1, -1)
                temporal_target_value_at_cur = dr.maximum(
                    sign * temporal_target_value_at_cur, 0)
            else:
                # Reroute samples so that all temporal samples that are positive are placed into
                # the lower half and negative are placed in the upper half
                index[temporal_target_value_at_cur >= 0] = \
                    index[temporal_target_value_at_cur >= 0] % self.n_params
                index[temporal_target_value_at_cur < 0] = \
                    index[temporal_target_value_at_cur < 0] % self.n_params + \
                    self.n_params

        # The target value, as well as W, comes pre-multiplied by the inverse of its candidate pdf
        # The product of these gives the jacobian determinant needed for the random replay shift
        reservoir.update(temporal_reservoir.is_light_sample,
                         dr.abs(temporal_target_value_at_cur) *
                         temporal_reservoir.W * temporal_mis_weight,
                         temporal_reservoir.sample_idx, temporal_sample_value_at_cur,
                         temporal_reservoir.uvw, sampler.next_1d(), index, temporal_reservoir.valid)
        reservoir.M += temporal_reservoir.M

        reservoir.finalize_resampling(
            dr.abs(self.ris_target_func(reservoir.sample_value)))

        self.temporal_reservoir = reservoir

        # ------ Update screen positions of each sample in the parameter-space reservoirs ---------
        # This is ugly but faster than performing within reservoir.update
        # It is possible for samples to be selected from 3 locations
        # 1. Current reservoir
        # 2. Temporal reservoir, with the same sign
        # 3. Temporal reservoir, with the opposite sign
        # Since there isn't an easy way for reservoir.update to determine this, we compare samples
        # and select the correct one
        selected_temporal_same_sign = dr.eq(reservoir.sample_idx, temporal_reservoir.sample_idx) & \
            dr.all(dr.eq(reservoir.uvw, temporal_reservoir.uvw))
        index_opp_sign = (dr.arange(
            mi.UInt32, self.get_reservoir_length()) + self.n_params) % self.n_params
        selected_temporal_opp_sign = \
            dr.eq(reservoir.sample_idx,
                  dr.gather(mi.UInt32, temporal_reservoir.sample_idx, index_opp_sign)) & \
            dr.all(dr.eq(reservoir.uvw, dr.gather(
                mi.Point3f, temporal_reservoir.uvw, index_opp_sign)))

        self.temporal_pos = dr.select(selected_temporal_same_sign,
                                      self.temporal_pos,
                                      dr.select(selected_temporal_opp_sign,
                                                dr.gather(
                                                    mi.Point2f, self.temporal_pos, index_opp_sign),
                                                cur_pos)
                                      )

        return reservoir

    def sample_ray_from_screen_pos(
            self,
            sensor: mi.Sensor,
            sampler: mi.Sampler,
            pos: mi.Point2f):

        film = sensor.film()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        return sensor.sample_ray_differential(
            time=time,
            sample1=wavelength_sample,
            sample2=pos_adjusted,
            sample3=aperture_sample
        )

    def compute_w(self,
                  pos: mi.Point2f,
                  grad_in: mi.TensorXf,
                  sensor: mi.Sensor,
                  spp: int,
                  weight_divide: bool = True,
                  active: mi.Bool = mi.Bool(True)):
        """
        This compute the weight w (Eqn 5 in paper), which is the sum of the product of the adjoint
        rendering and pixel filter for a sample x
        """
        film = sensor.film()
        film.prepare([])

        # Prepare an ImageBlock as specified by the film
        block = film.create_block(normalize=not weight_divide)

        # Only use the coalescing feature when rendering enough samples
        block.set_coalesce(block.coalesce() and spp >= 4)

        # Compute w: adjoint rendering and pixel filter
        with dr.resume_grad():
            L = dr.ones(mi.Color3f, dr.width(pos))
            dr.enable_grad(L)

            # Accumulate into the image block.
            block.put(
                pos=pos,
                wavelengths=dr.zeros(mi.Color0f, dr.width(pos)),
                value=L,
                active=active
            )

            film.put_block(block)

            # This step launches a kernel
            dr.schedule(block.tensor())
            image = film.develop(weight_divide=weight_divide)

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance
            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(mi.Float, dr.ADMode.Backward)

            # Make this independent of spp
            w = dr.grad(L) * spp

        del block
        return w

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)
            L, valid, state = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                si=si,
                depth=mi.UInt32(0),
                w=None,
                state_in=None,
                reparam=None,
                params=None,
                sample_idx=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = sensor.film().create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
                aovs = sensor.film().prepare_sample(L * weight, ray.wavelengths,
                                                    block.channel_count(), alpha=alpha)
                block.put(pos, aovs)
                del aovs
            else:
                block.put(pos, ray.wavelengths, L * weight, alpha)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid, alpha
            gc.collect()

            # Perform the weight division and return an image tensor
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()

            return self.primal_image

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        raise NotImplementedError("Only backward mode is implemented")

    def render_backward(self,
                        scene: mi.Scene,
                        params: mi.SceneParameters,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        if self.param_name is None:
            raise Exception('No param for reuse set via self.param_name!')

        film = sensor.film()
        aovs = self.aovs()

        # ReSTIR initialization
        if not self.use_ref and not self.restir_initialized:
            param_tensor: mi.TensorXf = params.get(self.param_name)

            self.n_params = param_tensor.shape[0] * param_tensor.shape[1]

            if self.param_name not in params.properties:
                raise Exception(f'Unknown texture {self.param_name}!')

            _, _, node, _ = params.properties[self.param_name]
            self.reuse_texture = node.id()

            # Initialize temporal buffers for first time
            if self.enable_temporal_reuse:
                self.temporal_param_tensor = param_tensor
                self.temporal_reservoir = mi.Reservoir(
                    self.get_reservoir_length())
                self.temporal_grad_in = grad_in
                self.temporal_pos = dr.zeros(
                    mi.Point2f, self.get_reservoir_length())

            self.restir_initialized = True

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos, _ = self.sample_rays(scene, sensor,
                                                   sampler)

            w = self.compute_w(pos, grad_in, sensor, spp)

            # Launch Monte Carlo sampling in backward AD mode (2)
            with dr.resume_grad():
                si: mi.SurfaceInteraction3f = scene.ray_intersect(ray)

            # ------------------------ Candidate generation (Alg 2 in paper) ----------------------
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                params=params,
                sampler=sampler,
                si=si,
                w=w,
                sample_idx=dr.arange(mi.UInt32, dr.prod(film.size()) * spp),
                active=mi.Bool(True)
            )

            if not self.use_ref:
                param_tensor: mi.TensorXf = params.get(self.param_name)

                # --------------- Iteration reuse in parameter-space (Alg 3 in paper) ------------
                if self.enable_temporal_reuse:
                    param_sampler = sampler.clone()
                    param_sampler.set_sample_count(1)
                    param_sampler.set_samples_per_wavefront(1)
                    param_sampler.seed(seed, self.get_reservoir_length())

                    valid = self.candidate_reservoir.valid
                    sample_idx = self.candidate_reservoir.sample_idx

                    # Get si and pos for all reservoirs
                    # Cheaper to retrace ray compared to gather from si buffer
                    ray = dr.gather(type(ray), ray, sample_idx, valid)
                    si = scene.ray_intersect(ray, valid)
                    pos = dr.gather(mi.Point2f, pos, sample_idx, valid)

                    reservoir = self.do_temporal_reuse(
                        scene, sensor, param_sampler, params, grad_in, si, pos)

                    # Update temporal buffers
                    self.temporal_grad_in = grad_in
                    self.temporal_param_tensor = param_tensor

                    # Materialize temporal data to prevent kernel recompilation
                    dr.eval(
                        self.temporal_pos,
                        self.temporal_grad_in,
                        self.temporal_param_tensor,
                        self.temporal_reservoir
                    )
                else:
                    reservoir = self.candidate_reservoir
                    reservoir.weight_sum /= reservoir.M
                    reservoir.finalize_resampling(
                        dr.abs(self.ris_target_func(reservoir.sample_value)))

                param_grad = reservoir.sample_value * reservoir.W

                # Align indices to account for param grad and radiance grad possibly
                # having different numbers of channels
                # Special case: if texture has only 1 channel,
                # convert 3 channel grad to 1 channel as well
                n_param_channel = param_tensor.shape[2]
                n_radiance_channel = dr.select(n_param_channel == 1, 1, 3)
                param_grad = dr.select(
                    n_param_channel == 1, dr.sum(param_grad), param_grad)
                channel_alignment = mi.Float(n_param_channel) / \
                    mi.Float(n_radiance_channel)

                # Convert array to tensor and update param grad
                if self.use_positivization:
                    # For positivization, we store the positive samples in the lower half
                    # and the negative samples in the upper half
                    index = dr.tile(
                        dr.arange(mi.Float, self.n_params) * channel_alignment, 2)
                else:
                    index = dr.arange(
                        mi.Float, self.n_params) * channel_alignment

                dr.scatter_reduce(
                    dr.ReduceOp.Add, self.param_tensor_grad.array, param_grad, index)
                with dr.resume_grad():
                    dr.set_grad(param_tensor, self.param_tensor_grad)

            # We don't need any of the outputs here
            del L, valid, state_out, w, \
                ray, weight, pos, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()


mi.register_integrator("restir_dr", lambda props: ReStirDRIntegrator(props))
