import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt
from pathlib import Path

@njit #avoids python overhead for this function
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0: #point escapes is outside mandelbrot set 
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit #avoids python overhead for this function
def mandelbrot_chunk(row_start, row_end, N,
                     x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)
    return out

# Computes the full N×N image by treating it as a single chunk.
def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args): #because pool.map requires a single argument
    return mandelbrot_chunk(*args)

if __name__ == "__main__":
    N = 1024
    max_iter = 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter) #in order to not include compilation time 

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times) #to reduce effect of outliers 

    cpu_count = os.cpu_count()

    for n_workers in range(1, cpu_count + 1):
        chunk_size = max(1, N // n_workers)
        chunks = []
        row = 0
        while row < N:
            end = min(row + chunk_size, N)
            chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
            row = end

        with Pool(processes=n_workers) as pool:
            pool.map(_worker, chunks)

        times = []
        with Pool(processes=n_workers) as pool:
            for _ in range(3):
                t0 = time.perf_counter()
                result = np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
        t_par = statistics.median(times)
        speedup = t_serial / t_par
        efficiency = (speedup / n_workers) * 100
        print(f"{n_workers:2d} workers : {t_par:.3f} s, "
              f"speedup = {speedup:.2f}x, eff = {efficiency:.0f}%")

    final_image = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    plt.imshow(final_image, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
               origin='lower', cmap='hot')
    plt.colorbar(label='Iteration count')
    plt.title(f"Mandelbrot set ({N}×{N}, max_iter={max_iter})")
    plt.show()