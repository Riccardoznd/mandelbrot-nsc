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

@njit(cache=True)
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

@njit(cache=True)
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
    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    p_workers = 8
    
    client = Client("tcp://10.92.1.162:8786")
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX,
                                        Y_MIN, Y_MAX, 10))

    chunk_sizes = [16, 32, 64, 128, 256, 512]
    results_data = []
    
    for n_chunks in chunk_sizes:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, 
                                    max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)
        
        median_time = statistics.median(times)
        print(f"n_chunks={n_chunks}: {median_time:.3f} s")
        results_data.append((n_chunks, median_time))
    
    cluster_single = LocalCluster(n_workers=1, threads_per_worker=1)
    client_single = Client(cluster_single)
    client_single.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX,
                                               Y_MIN, Y_MAX, 10))
    t1_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, 
                           max_iter, n_chunks=32)
        t1_times.append(time.perf_counter() - t0)
    T1 = statistics.median(t1_times)
    client_single.close()
    cluster_single.close()
    
    lif_data = []
    for n_chunks, Tp in results_data:
        lif = p_workers * Tp / T1 - 1
        lif_data.append((n_chunks, lif))
    
    optimal_idx = np.argmin([r[1] for r in results_data])
    n_chunks_optimal = results_data[optimal_idx][0]
    t_min = results_data[optimal_idx][1]
    LIF_min = lif_data[optimal_idx][1]
    print(f"n_chunks_optimal={n_chunks_optimal}, t_min={t_min:.3f} s, LIF_min={LIF_min:.3f}")
    
    result_optimal = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, 
                                     max_iter, n_chunks=n_chunks_optimal)
    plt.figure(figsize=(8, 8))
    plt.imshow(result_optimal, extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap='hot', origin='lower')
    plt.savefig(f'mandelbrot_dask_chunks{n_chunks_optimal}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    chunk_sizes_plot = [r[0] for r in results_data]
    times_plot = [r[1] for r in results_data]
    
    plt.figure(figsize=(8, 5))
    plt.plot(chunk_sizes_plot, times_plot, marker='o', linewidth=2)
    plt.xscale('log')
    plt.xlabel('n_chunks (log scale)')
    plt.ylabel('Wall time (s)')
    plt.title('Dask Local: Chunk Size Sweep')
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig('dask_chunk_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    client.close()