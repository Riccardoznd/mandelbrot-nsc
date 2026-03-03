"""
Mandelbrot Set Generator numba version

Author: Riccardo Zanda
Course: Numerical Scientific Computing 2026
"""
import matplotlib.pyplot as plt
import time
import statistics
import numpy as np
from numba import njit

#benchmark fucntion as per slides
def bench(fn, *args, runs=5):
    fn(*args)  # extra warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)

#define region boundaries
x_min=-2.0
x_max=1.0
y_min=-1.5
y_max=1.5

#define resolution
width=4096
height=4096
max_iter=100

#approach A

@njit
def mandelbrot_point_numba(c, max_iter):
    z = 0j  # Complex literal required for type inference (Slide 35)
    for n in range(max_iter):
        # Condition must be on one line (Slide 34)
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iter #to check whether everithing worked out correctly

def mandelbrot_hybrid(xmin, xmax, ymin, ymax, width, height, max_iter):
    #equally spaced vectors
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    # createsn 2D array
    result = np.zeros((height, width), dtype=np.int32)
    
   #outer loop still in python
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            result[i, j] = mandelbrot_point_numba(c, max_iter)
    return result

#approach B (suggested)
@njit
def mandelbrot_naive_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0j
            n = 0
            while n < max_iter and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z * z + c
                n += 1
            result[i, j] = n
    return result

# Warm up (triggers JIT compilation--exclude from timing)
Warm_upA= mandelbrot_hybrid(-2, 1, -1.5, 1.5, 64, 64)
Warm_upB= mandelbrot_naive_numba(-2, 1, -1.5, 1.5, 64, 64)

t_hybrid = bench(mandelbrot_hybrid, -2, 1, -1.5, 1.5, 1024, 1024)
t_full = bench(mandelbrot_naive_numba, -2, 1, -1.5, 1.5, 1024, 1024)

print(f"Hybrid: {t_hybrid:.3f}s")
print(f"Fully compiled: {t_full:.3f}s")
print(f"Ratio: {t_hybrid/t_full:.1f}x")
#actual
plt.figure(figsize=(10, 10))
plt.imshow(Warm_up, extent=[-2, 1, -1.5, 1.5], origin='lower', cmap='hot')
plt.colorbar()
plt.show()

