#pragma once

#include <drjit/array.h>
#include <drjit/struct.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Parameter-space Reservoir (Alg 1 in paper)
 */
template <typename Float_>
class Reservoir {
public:
    // Types
    using Float                       = Float_;

    MI_IMPORT_CORE_TYPES();

    // Constructors
    Reservoir(uint32_t N)
        : m_weight_sum(dr::zeros<Float>(N)),
          m_M(0),
          m_lock(dr::zeros<UInt32>(N)),
          m_sample_idx(dr::zeros<UInt32>(N)),
          m_sample_value(dr::zeros<Vector3f>(N)),
          m_uvw(dr::zeros<Point3f>(N)) {}

    // Reservoir functionality
    void update(const Bool& is_light_sample, const Float& weight, const UInt32& sample_idx,
                const Vector3f& sample_value, const Point3f& uvw,
                const Float& random_num, const UInt32& index, const Mask& mask = true) {
        if constexpr (dr::is_array_v<Float>) {
            // Packed thread idx and weight_sum, used to reduce lock contention
            UInt64 weight_sum_index = UInt64(dr::reinterpret_array<UInt32>(m_weight_sum));
            ReservoirSampleData reservoir_data = {
                weight_sum_index.index(),
                m_sample_idx.index(),
                m_uvw.entry(0).index(),
                m_uvw.entry(1).index(),
                m_uvw.entry(2).index(),
                m_sample_value.entry(0).index(),
                m_sample_value.entry(1).index(),
                m_sample_value.entry(2).index()
            };
            uint32_t lock_index = m_lock.index();

            // light sample denoted by -1 for w
            Point3f uvw_is_light = uvw;
            uvw_is_light.entry(2) = dr::select(is_light_sample, -1, uvw_is_light.entry(2));
            ReservoirSampleData sample_data = {
                weight.index(),
                sample_idx.index(),
                uvw_is_light.entry(0).index(),
                uvw_is_light.entry(1).index(),
                uvw_is_light.entry(2).index(),
                sample_value.entry(0).index(),
                sample_value.entry(1).index(),
                sample_value.entry(2).index()
            };

            // Special function to atomically update weight_sum and select sample
            jit_var_new_reservoir_sampling(reservoir_data, lock_index, sample_data,
                                           random_num.index(), index.index(), mask.index());

            // Update member variables with jit output
            m_lock = UInt32::steal(lock_index);
            m_weight_sum = dr::reinterpret_array<Float>(
                UInt32(UInt64::steal(reservoir_data.weight)));
            m_sample_idx = UInt32::steal(reservoir_data.sample_idx);
            m_uvw.entry(0) = Float::steal(reservoir_data.u);
            m_uvw.entry(1) = Float::steal(reservoir_data.v);
            m_uvw.entry(2) = Float::steal(reservoir_data.w);
            m_sample_value.entry(0) = Float::steal(reservoir_data.value_r);
            m_sample_value.entry(1) = Float::steal(reservoir_data.value_g);
            m_sample_value.entry(2) = Float::steal(reservoir_data.value_b);

            // No efficient way to update M, so do it outside this class
        } else {
            Throw("Reservoir::update for scalars not implemented");
        }
    }

    void finalize_resampling(const Float& target_value) {
        if constexpr (dr::is_array_v<Float>) {
            m_weight_sum = dr::select(dr::eq(target_value, 0), 0, m_weight_sum / target_value);
        } else {
            Throw("Reservoir::update for scalars not implemented");
        }
    }

    Bool is_valid() const {
        return dr::neq(m_weight_sum, 0);
    }

    Bool is_light_sample() const {
        return m_uvw.entry(2) < 0;
    }

    Float m_weight_sum;               /// Sum of weights in reservoir, overloaded with
                                      /// the unbiased contribution weight W
    Float m_M;                        /// Number of candidates
    UInt32 m_lock;                    /// Reservoir lock
    UInt32 m_sample_idx;              /// Index of sample
    Vector3f m_sample_value;          /// Sample value
    Point3f m_uvw;                    /// Sample random numbers
};

/// Return a string representation of the ray
template <typename Float>
std::ostream &operator<<(std::ostream &os, const Reservoir<Float> &r) {
    os << "Reservoir " << "[" << std::endl
       << "  weight_sum = " << r.m_weight_sum << "," << std::endl
       << "  M = " << r.m_M << "," << std::endl
       << "  sample_idx = " << r.m_sample_idx << "," << std::endl
       << "  sample_value = " << r.m_sample_value << "," << std::endl
       << "  uvw = " << r.m_uvw << "," << std::endl
       << "]";
    return os;
}

NAMESPACE_END(mitsuba)
