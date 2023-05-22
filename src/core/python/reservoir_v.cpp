#include <mitsuba/core/reservoir.h>
#include <mitsuba/python/python.h>

MI_PY_EXPORT(Reservoir) {
    MI_PY_IMPORT_TYPES(Float)

    using Reservoir = Reservoir<Float>;

    auto reservoir = py::class_<Reservoir>(m, "Reservoir", D(Reservoir))
        .def(py::init<uint32_t>(), "N"_a)
        .def("update", &Reservoir::update,
            "is_light_sample"_a, "weight"_a, "sample_idx"_a,
            "sample_value"_a, "uvw"_a,
            "random_num"_a, "index"_a, "mask"_a = true)
        .def("finalize_resampling", &Reservoir::finalize_resampling, "target_value"_a)
        .def_readwrite("weight_sum", &Reservoir::m_weight_sum, D(Reservoir, weight_sum))
        .def_readwrite("M", &Reservoir::m_M, D(Reservoir, M))
        .def_readwrite("sample_idx", &Reservoir::m_sample_idx, D(Reservoir, sample_idx))
        .def_readwrite("sample_value", &Reservoir::m_sample_value, D(Reservoir, sample_value))
        .def_readwrite("uvw", &Reservoir::m_uvw, D(Reservoir, uvw))
        .def_readwrite("W", &Reservoir::m_weight_sum, D(Reservoir, W))
        .def_property_readonly("valid", &Reservoir::is_valid)
        .def_property_readonly("is_light_sample", &Reservoir::is_light_sample)
        .def_repr(Reservoir);
}
