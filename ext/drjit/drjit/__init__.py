import sys as _sys
import os as _os

if _sys.version_info < (3, 8):
    raise ImportError("Dr.Jit requires Python >= 3.8")

# Implementation details accessed by both C++ and Python
import drjit.detail as detail # noqa

# Configuration information generated by CMake
import drjit.config as _config

# Context manager to temporarily use RTLD_DEEPBIND with Clang on Linux to
# prevent the DLL to search symbols in the global scope.
class scoped_rtld_deepbind:
    def __init__(self) -> None:
        # It is possible to set the DRJIT_NO_RTLD_DEEPBIND environement variable
        # to prevent Dr.Jit to use RTLD_DEEPBIND. This is necessary when using
        # Dr.Jit's shared library from an executable that is not compiled with
        # -fPIC. More information regarding this issue can be found here:
        # https://codeutility.org/linux-weird-interaction-of-rtld_deepbind-position-independent-code-pic-and-c-stl-stdcout-stack-overflow/
        self.cond = _sys.platform != 'darwin' and _os.name != 'nt' \
                    and not 'DRJIT_NO_RTLD_DEEPBIND' in _os.environ \
                    and 'Clang' in _config.CXX_COMPILER

    def __enter__(self):
        if self.cond:
            self.backup = _sys.getdlopenflags()
            _sys.setdlopenflags(_os.RTLD_LAZY | _os.RTLD_LOCAL | _os.RTLD_DEEPBIND)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cond:
            _sys.setdlopenflags(self.backup)

# Native extension defining low-level arrays
with scoped_rtld_deepbind():
    import drjit.drjit_ext as drjit_ext  # noqa

# Routing functionality (type promotion, broadcasting, etc.)
import drjit.router as router  # noqa

# Generic fallback implementations of array operations
import drjit.generic as generic  # noqa

# Type traits analogous to the ones provided in C++
import drjit.traits as traits  # noqa

# Math library and const
import drjit.const as const  # noqa

# Matrix-related functions
import drjit.matrix as matrix # noqa

# Tensor-related functions
import drjit.tensor as tensor # noqa

# Install routing functions in ArrayBase and global scope
self = vars()
base = self['ArrayBase']
for k, v in router.__dict__.items():
    if k.startswith('_') or (k[0].isupper() and not k == 'CustomOp'):
        continue
    if k.startswith('op_'):
        setattr(base, '__' + k[3:] + '__', v)
    else:
        self[k] = v

# Install generic array functions in ArrayBase
for k, v in generic.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    if k.startswith('op_'):
        setattr(base, '__' + k[3:] + '__', v)
    else:
        setattr(base, k, v)


# Install type traits in global scope
for k, v in traits.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


# Install constants in global scope
for k, v in const.__dict__.items():
    if k.startswith('_'):
        continue
    self[k] = v


# Install matrix-related functions in global scope
for k, v in matrix.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v


# Install tensor-related functions
for k, v in tensor.__dict__.items():
    if k.startswith('_') or k[0].isupper():
        continue
    self[k] = v

del k, v, self, base, generic, router, matrix, tensor, traits, const, drjit_ext
