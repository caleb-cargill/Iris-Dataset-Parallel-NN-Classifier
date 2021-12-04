/*
 CS5168/CS6068 - Parallel Computing

 Final Project - Iris Dataset Parallel NN Classifier
 November, 2021

 Team Members:
 - Ben Elfner
 - Caleb Cargill

 File Description:
 - used for tracking execution time
*/

#include <driver_types.h>
#include <cuda_runtime.h>
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

#endif  /* __GPU_TIMER_H__ */

