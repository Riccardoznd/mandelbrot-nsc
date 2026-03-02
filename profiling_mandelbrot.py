import cProfile
import pstats
from naivemandelbrot import naive_mandelbrot
from mandelbrot import my_mandelbrot

print("PROFILING NAIVE AND NUMPY VERSIONS")

cProfile.run('naive_mandelbrot(-2, 1, -1.5, 1.5, 512, 512, 100)','naive_profile.prof')
print("NAIVE VERSION :\n")
stats_naive = pstats.Stats('naive_profile.prof')
stats_naive.sort_stats('cumulative')
stats_naive.print_stats(10)

cProfile.run('my_mandelbrot(-2, 1, -1.5, 1.5, 512, 512, 100)','numpy_profile.prof')
print("NUMPY VERSION: \n")

stats_numpy = pstats.Stats('numpy_profile.prof')
stats_numpy.sort_stats('cumulative')
stats_numpy.print_stats(10)
