/*
    tests/conv.cpp -- tests special functions

    Dr.Jit is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <drjit/special.h>

DRJIT_TEST_FLOAT(test01_i0e)  {
    using Scalar = scalar_t<T>;

    double results[] = {
        1.000000000,  0.4657596076, 0.3085083226, 0.2430003542,
        0.2070019212, 0.1835408126, 0.1666574326, 0.1537377447,
        0.1434317819, 0.1349595246, 0.1278333372, 0.1217301682,
        0.1164262212, 0.1117608338, 0.1076152517, 0.1038995314
    };

    for (int i = 0; i < 16; ++i)
        assert(max(abs(i0e(T(Scalar(i))) - T(Scalar(results[i])))) < 1e-6);
}

DRJIT_TEST_FLOAT(test02_erf) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return erf(a); },
        [](double a) { return std::erf(a); },
        Value(-1), Value(1), 6
    );

    Array<T, 4> x((Value) 0.5);
    Array<T&, 4> y(x);
    assert(erf(x) == erf(y));
}

DRJIT_TEST_FLOAT(test02_erfc) {
    test::probe_accuracy<T>(
        [](const T &a) -> T { return erfc(a); },
        [](double a) { return std::erfc(a); },
        Value(-1), Value(1), 17
    );

    Array<T, 4> x((Value) 0.5);
    Array<T&, 4> y(x);
    assert(erfc(x) == erfc(y));
}

DRJIT_TEST_FLOAT(test04_erfinv) {
    for (int i = 0; i < 1000; ++i) {
        auto f = T((float) i / 1000.0f * 2 - 1 + 1e-6f);
        auto inv = erfinv(f);
        auto f2 = erf(inv);
        assert(std::abs(T(f-f2)[0]) < 1e-6f);
    }
}

DRJIT_TEST_FLOAT(test05_dawson)  {
    using Scalar = scalar_t<T>;

    double results[] = { 0.0,
        0.09933599239785286, 0.1947510333680280, 0.2826316650213119,
        0.3599434819348881, 0.4244363835020223, 0.4747632036629779,
        0.5105040575592318, 0.5321017070563654, 0.5407243187262987,
        0.5380795069127684, 0.5262066799705525, 0.5072734964077396,
        0.4833975173848241, 0.4565072375268973, 0.4282490710853986,
        0.3999398943230814, 0.3725593489740788, 0.3467727691148722,
        0.3229743193228178, 0.3013403889237920, 0.2818849389255278,
        0.2645107599508320, 0.2490529568377667, 0.2353130556638426,
        0.2230837221674355, 0.2121651242424990, 0.2023745109105140,
        0.1935507238593668, 0.1855552345354998, 0.1782710306105583 };


    for (int i = 0; i <= 30; ++i) {
        assert(max(abs(dawson(T(Scalar(i * 0.1)))  - T(Scalar( results[i])))) < 1e-6);
        assert(max(abs(dawson(T(Scalar(i * -0.1))) - T(Scalar(-results[i])))) < 1e-6);
    }
}

DRJIT_TEST_FLOAT(test06_ellint_1) {
    double result[] = { -14.28566868, -13.24785552, -11.73287659, -9.893205471,
        -8.835904570, -7.468259577, -5.516159896, -4.419121129, -3.170330916,
        -1.159661071, 0, 1.159661071, 3.170330916, 4.419121129, 5.516159896,
        7.468259577, 8.835904570, 9.893205471, 11.73287659, 13.24785552,
        14.28566868 };

    for (int i=-10; i<=10; ++i)
        assert((ellint_1(T((float) i), T(.9f))[0] - result[i+10]) < 2e-6);
}

DRJIT_TEST_FLOAT(test07_comp_ellint_1) {
    double result[] = { 1.570796327, 1.574745562, 1.586867847, 1.608048620,
                        1.639999866, 1.685750355, 1.750753803, 1.845693998,
                        1.995302778, 2.280549138 };

    for (int i=0; i<10; ++i)
        assert((comp_ellint_1(T((float) i/10.f))[0] - result[i]) < 1e-6);
}

DRJIT_TEST_FLOAT(test08_ellint_2) {
    double result[] = { -7.580388582, -6.615603622, -5.923080706, -5.355941680,
        -4.406649345, -3.647370231, -3.121560380, -2.202184075, -1.380263348,
        -0.8762622200, 0, 0.8762622200, 1.380263348, 2.202184075, 3.121560380,
        3.647370231, 4.406649345, 5.355941680, 5.923080706, 6.615603622,
        7.580388582 };

    for (int i=-10; i<=10; ++i)
        assert((ellint_2(scalar_t<T>(i), T(.9f))[0] - result[i+10]) < 3e-6);
}

DRJIT_TEST_FLOAT(test09_comp_ellint_2) {
    double result[] = { 1.570796327, 1.566861942, 1.554968546, 1.534833465,
                        1.505941612, 1.467462209, 1.418083394, 1.355661136,
                        1.276349943, 1.171697053 };

    for (int i=0; i<10; ++i)
        assert((comp_ellint_2(T((float) i/10.f))[0] - result[i]) < 1e-6);
}

DRJIT_TEST_FLOAT(test10_ellint_3) {
    double values[] = {
        1.000000000,  1.001368050,  1.005529262,  1.012662720,  1.023095616,
        1.037356120,  1.056273110,  1.081169466,  1.114267715,  1.159661071,
        0.9739108232, 0.9752197204, 0.9792006242, 0.9860236410, 0.9959994565,
        1.009629332,  1.027699365,  1.051463019,  1.083023814,  1.126250332,
        0.9499559681, 0.9512112978, 0.9550289363, 0.9615709238, 0.9711331194,
        0.9841926334, 1.001497200,  1.024238112,  1.054412465,  1.095688359,
        0.9278496365, 0.9290561820, 0.9327251506, 0.9390112903, 0.9481970619,
        0.9607377650, 0.9773465184, 0.9991585837, 1.028075279,  1.067584596,
        0.9073578181, 0.9085197125, 0.9120526214, 0.9181046645, 0.9269461292,
        0.9390125110, 0.9549855234, 0.9759496456, 1.003719507,  1.041620361,
        0.8882866913, 0.8894075341, 0.8928153640, 0.8986522470, 0.9071773565,
        0.9188081175, 0.9341976099, 0.9543840576, 0.9811032294, 1.017532566,
        0.8704740713, 0.8715570194, 0.8748493996, 0.8804877250, 0.8887209781,
        0.8999500224, 0.9148017149, 0.9342719585, 0.9600244531, 0.9951017555,
        0.8537829977, 0.8548308371, 0.8580162656, 0.8634706822, 0.8714336878,
        0.8822909075, 0.8966450863, 0.9154532476, 0.9403129557, 0.9741431606,
        0.8380968526, 0.8391120566, 0.8421980757, 0.8474815829, 0.8551935114,
        0.8657054139, 0.8795977838, 0.8977917933, 0.9218240876, 0.9544999068,
        0.8233155947, 0.8243003700, 0.8272937093, 0.8324179034, 0.8398958507,
        0.8500860662, 0.8635484385, 0.8811709703, 0.9044339982, 0.9360377889
    };

    int k = 0;
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i)
            assert(std::abs(ellint_3(1.0f, (float) i / 10.f, T((float) j / 10.f))[0] - values[k++]) <
                   1e-6f);

    double values2[] = { -11.3057, -10.3096, -9.16461, -7.87078, -6.87262,
                         -5.78783, -4.44026, -3.4361,  -2.39321, -1.01753,
                         0,        1.01753,  2.39321,  3.4361,   4.44026,
                         5.78783,  6.87262,  7.87078,  9.16461,  10.3096,
                         11.3057 };

    k = 0;
    for (int i = -10; i <= 10; ++i)
        assert(std::abs(ellint_3(T((float) i), 0.9f, 0.5f)[0] - values2[k++]) < 5e-5f);
}

DRJIT_TEST_FLOAT(test11_comp_ellint_3) {
    double values[] = {
        1.570796327, 1.574745562, 1.586867847, 1.608048620, 1.639999866,
        1.685750355, 1.750753803, 1.845693998, 1.995302778, 2.280549138,
        1.497695533, 1.501371111, 1.512651347, 1.532353469, 1.562056689,
        1.604552494, 1.664861577, 1.752805017, 1.891075542, 2.153786851,
        1.433934302, 1.437374939, 1.447932393, 1.466365815, 1.494141434,
        1.533849048, 1.590141802, 1.672109878, 1.800722666, 2.044319458,
        1.377679515, 1.380915961, 1.390845351, 1.408176743, 1.434278986,
        1.471568194, 1.524381424, 1.601181365, 1.721461105, 1.948628026,
        1.327565199, 1.330622327, 1.340000252, 1.356364354, 1.380998621,
        1.416167952, 1.465934528, 1.538216200, 1.651226784, 1.864111423,
        1.282549830, 1.285448071, 1.294337440, 1.309844876, 1.333179718,
        1.366473953, 1.413548429, 1.481843319, 1.588452895, 1.788801324,
        1.241823533, 1.244579894, 1.253033068, 1.267775880, 1.289951467,
        1.321574029, 1.366250754, 1.430999474, 1.531926255, 1.721178113,
        1.204745787, 1.207374591, 1.215435656, 1.229491324, 1.250625592,
        1.280747518, 1.323273747, 1.384845919, 1.480691232, 1.660048075,
        1.170802455, 1.173315887, 1.181022345, 1.194456757, 1.214649957,
        1.243416541, 1.284002126, 1.342711065, 1.433983702, 1.604459196,
        1.139575429, 1.141983949, 1.149367992, 1.162237690, 1.181575812,
        1.209111610, 1.247936297, 1.304050050, 1.391184541, 1.553642024
    };

    int k = 0;
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i)
            assert(std::abs(comp_ellint_3((double) i / 10.0, T((float) j / 10.f))[0] - values[k++]) <
                   1e-6f);
}
