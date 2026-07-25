#! /usr/bin/env python3

import ctypes
import os
import time
import numpy as np

class FourierEngineWrapper:
    """High-performance Python wrapper interface binding the native C

    Fourier engine using ctypes and contiguous double-complex pointers.
    """

    def __init__(self, lib_path: str = "./libfourier.so") -> None:
        # Load the binary dynamic shared object pointer
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Shared object binary not found: {lib_path}"
            )

        self._lib = ctypes.CDLL(os.path.abspath(lib_path))

        # Explicitly configure the native C function signature layout
        self._lib.fast_fourier_integral_c99.argtypes = [
            ctypes.c_double,  # x0
            ctypes.c_double,  # x1
            ctypes.c_double,  # y0
            ctypes.c_double,  # y1
            ctypes.c_double,  # u
            ctypes.c_double,  # v
            ctypes.c_int,  # max_n
            ctypes.c_int,  # num_threads
        ]

        # Returns a pointer to raw double-precision complex elements
        self._lib.fast_fourier_integral_c99.restype = (
            ctypes.POINTER(ctypes.c_double * 2)
        )

    def integrate(
        self,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        u: float,
        v: float,
        max_n: int,
        num_threads: int = 4,
    ) -> np.ndarray:
        """Evaluates the exact analytical Fourier integral on C threads.

        Returns a 1D NumPy array of shape (max_n,) with complex128 data.
        """
        if max_n <= 0:
            return np.empty(0, dtype=np.complex128)

        # Call the bare-metal binary library routine directly
        raw_ptr = self._lib.fast_fourier_integral_c99(
            float(x0),
            float(x1),
            float(y0),
            float(y1),
            float(u),
            float(v),
            int(max_n),
            int(num_threads),
        )

        if not raw_ptr:
            raise MemoryError("C allocation layer failed to return memory.")

        # Cast the pointer block to a managed float64 contiguous matrix layout
        buffer_size = max_n * 2
        float_array = ctypes.cast(
            raw_ptr, ctypes.POINTER(ctypes.c_double * buffer_size)
        ).contents

        # Re-interpret the memory block into a native NumPy array views
        np_floats = np.frombuffer(float_array, dtype=np.float64)

        # View interleaved real/imaginary vectors as double complex numbers
        complex_result = np_floats.view(np.complex128)

        # Create an independent copy before manually freeing C memory
        python_owned_array = complex_result.copy()

        # Free the C memory buffer immediately to prevent leaks
        ctypes.CDLL(None).free(raw_ptr)

        return python_owned_array


# --- Verification & Throughput Execution Profile ---
if __name__ == "__main__":
    # Specify your path context parameters
    target_lib = "./libfourier.so" if os.name != "nt" else "./fourier.dll"

    try:
        engine = FourierEngineWrapper(lib_path=target_lib)

        params = {
            "x0": 1.2,
            "x1": 4.7,
            "y0": -2.0,
            "y1": 10.5,
            "u": 0.65,
            "v": -0.45,
            "max_n": 200000,
            "num_threads": 4,
        }
        #print("Executing bare-metal C via Python wrapper loop...")
        mn = 1000000
        for n in range(15):

            start_time = time.perf_counter()
            results = engine.integrate(**params)
            end_time = time.perf_counter()

            #print(f"Processed array vector shape: {results.shape}")
            t = (end_time - start_time) * 1000
            print(f"Time: {t:.3f} ms")
            mn = min(t,mn)
            #print(f"Index 0 (n=0): {results[0]:.12f}")
            #print(f"Index 5 (n=5): {results[5]:.12f}")
        print(f"min: {mn:.3f} ms")
    except Exception as error:
        print(f"Execution failed: {error}")
        print("Verify the shared library path name is compiled correctly.")
