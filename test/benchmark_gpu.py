# PyVot
# Author: Liang Mi <icemiliang@gmail.com>
# Date: Jan 21st 2019

"""
=============================
        Running time
=============================

The running time of area preserving mapping using numpy and cupy

"""

from vot import *
from vot_gpu import *
import time

def test_area_preserve(N):
    print("-------\nnum_p: "+"{:,}".format(60)+", num_e: "+"{:,}".format(60*N))
    # ----- cpu ----- #
    ot = VotAreaPreserve()
    ot.import_data_from_file('../demo/data/p.csv')
    ot.setup(max_iter=3000, ratio=N, rate=0.2, dim=2, verbose=False)
    tick = time.clock()
    ot.area_preserve()
    tock = time.clock()
    print("CPU: " + str(tock - tick))

    # ----- gpu ----- #
    ot_gpu = VotAreaPreserveGPU()
    ot_gpu.import_data_from_file("../demo/data/p.csv")
    ot_gpu.setup(max_iter=3000, ratio=N, rate=0.2, dim=2, verbose=False)
    tick = time.clock()
    ot_gpu.area_preserve()
    tock = time.clock()
    print("GPU: " + str(tock - tick))
# ----------------------------------- #
#
# The following tests will take about 5 - 10 minutes.
#
# The tests will show that larger size does not add too much burden
# to the GPU solver. CPU and GPU versions spend similar time when N = 900.

# ----------------------------------- #
print("\n--> Testing area-preserving mapping with CPU(NumPy) and GPU(CuPy)\n")
# 60 x 60 x 200 = 720,000
N = 200
test_area_preserve(N)

# 60 x 60 x 500 = 1,960,000
N = 500
test_area_preserve(N)

# 60 x 60 x 1000 = 3,600,000
N = 1000
test_area_preserve(N)

# 60 x 60 x 2000 = 7,200,000
N = 2000
test_area_preserve(N)
