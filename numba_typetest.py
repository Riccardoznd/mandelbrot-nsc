"""
Mandelbrot Set Generator-type testing

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026
"""

from numba import njit
import numpy as np, time
import matplotlib.pyplot as plt

@njit
def mandelbrot_point_numba(c, max_iter):
    z = 0j
    for n in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter

@njit
def mandelbrot_numba_typed(xmin, xmax, ymin, ymax,
width, height, max_iter=100, dtype=np.float64):
    x = np.linspace(xmin, xmax, width).astype(dtype)
    y = np.linspace(ymin, ymax, height).astype(dtype)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result

# Benchmark dtypes
for dtype in [np.float32, np.float64]:
    # Warm-up is important
    warm_up = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 64, 64, 100, dtype=dtype)
    
    # starting the timer 
    t0 = time.perf_counter()
    result = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=dtype)
    t = time.perf_counter() - t0
    print(f"{dtype.__name__:8s}: {t:.3f}s")

# Generating images for visual comparison
r32 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=np.float32)
r64 = mandelbrot_numba_typed(-2, 1, -1.5, 1.5, 1024, 1024, 100, dtype=np.float64)

# Side-by-side plot in order to compare them 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
results = [r32, r64]
titles = ['float32', 'float64 (ref)']

for ax, result, title in zip(axes, results, titles):
    ax.imshow(result, cmap='hot')
    ax.set_title(title)
    ax.axis('off')

plt.savefig('precision_comparison.png', dpi=150)  # Saves the file
plt.show()                                        # Opens the window

# Precision analysis
print(f"Max diff float32 vs float64: {np.abs(r32 - r64).max()}")
print(f"Different pixels: {np.sum(r32 != r64)}")