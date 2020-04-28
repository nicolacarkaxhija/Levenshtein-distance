#include "BarrierEditDistance.h"

int BarrierEditDistance::numThreads = 0;
barrier BarrierEditDistance::diagonalBarrier(MAX_NUM_THREAD);
