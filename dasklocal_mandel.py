import numpy as np
from numba import njit
from multiprocessing import Pool
import time
import os
import statistics
import matplotlib.pyplot as plt
from pathlib import Path
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster

@njit(cache=True) #avoids python overhead for this function
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real * z_real
        zi2 = z_imag * z_imag
        if zr2 + zi2 > 4.0:
            return i
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)# cache=True: saves compiled code to disk so workers load instead of re-compiling
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

def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                    max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == '__main__':
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX,
                                        Y_MIN, Y_MAX, 10))

chunk_sizes = [16, 32, 64, 128, 256, 512]
    results_data = []  # Store (n_chunks, median_time) for plotting
    
    for n_chunks in chunk_sizes:
        times = []
        for _ in range(3):  # 3 runs for median
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, 
                                    max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        
        median_time = statistics.median(times)
        print(f"n_chunks={n_chunks}: {median_time:.3f} s")
        results_data.append((n_chunks, median_time))
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    print(f"Dask local(n_chunks=32): {statistics.median(times):.3f} s")
    client.close()
    cluster.close()

plt.figure(figsize=(8, 8))
plt.imshow(result, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap='hot', origin='lower')
plt.close()