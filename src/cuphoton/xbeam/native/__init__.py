# SPDX-License-Identifier: Apache-2.0
"""Optional native radio execution with a thin Cython interface."""


def cuda_available() -> bool:
    """Whether the optional extension includes native CUDA execution."""
    try:
        from ._binding import cuda_available as available
    except ModuleNotFoundError as error:
        if error.name != __name__ + "._binding":
            raise
        return False
    return available()


def correlate(*args, **kwargs):
    """Run native radio IO, planning, CUDA execution, and SWIN writing.

    The optional extension releases the GIL for the complete C++ run. It
    takes file paths and a bounded scan/integration scope; no Python job or
    GPU-array objects participate in native execution.
    """
    from ._binding import correlate as execute

    return execute(*args, **kwargs)
