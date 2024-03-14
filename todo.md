# chapter 0 - introduction

1. clock - calculates time taken to process a block of threads
1. clock_nvrtc - skip
1. simpleOccupancy
1. simpleP2P - skip, needs multiple GPUs
1. simplePrintf - shows use of printf from the kernel
1. asyncAPI - shows use of cudaMemcpyAsync from host to device and device to host
1. c++11_cuda - not that useful, can skip
1. fp16ScalarProduct - calculates scalar product of two vectors using native fp16 and intrinsics
1. cppIntegration - integrates a cuda kernel with c++ code. 
1. cppOverload
1. simpleCallback - skip
1. simpleDrvRuntime - shows use of driver API
1. simpleIPC
1. simpleSeparateCompilation - shows use of cudaMemcpyFromSymbol

## copy

1. simpleZeroCopy
1. simpleMultiCopy

## streams

1. simpleStreams
1. concurrentKernels
1. UnifiedMemoryStreams
1. simpleAttributes - ??

## vector addition

1. vectorAdd
1. vectorAddMMAP

## matmul

1. matrixMul

## assert

1. simpleAssert - shows use of assert statement from c++

## atomic intrinsics

1. simpleAtomicIntrinsics
1. systemWideAtomics

## vote intrinsics

1. simpleVoteIntrinsics

## templates

1. template
1. simpleTemplates

## cooperative groups

1. simpleCooperativeGroups
1. simpleAWBarrier

# chapter 1 - utilities

1. bandwidthTest
1. deviceQuery
1. topologyQuery

# chapter 2 - concepts

1. cuHook
1. FunctionPointers
1. histogram
1. imlinePTX
1. interval
1. reduction
1. reductionMultiBlockCG
1. scalarProd
1. scan
1. shfl scan

(optional)
1. streamOrderedAllocation*


# chapter 3 - features

## tensor cores
- bf16TensorCoreGemm
- cudaTensorCoreGemm
- dmmaTensorCoreGemm
- immaTensorCoreGemm
- tf32TensorCoreGemm

## cuda dynamic parallelism

- cdpSimplePrint
- cdpQuickSort
- cdpAdvancedQuickSort

## cooperative groups
- binaryPartitionCG
- warpAggregatedAtomicsCG

## cuda graphs
- simpleCudaGraphs
- graphMemoryNodes
- graphMemoryFootprint

## misc
- globalToShmemAsyncCopy
- streamPriorities

# chapter 4 - cuda libraries

- conjugateGradient*
- matMulCUBLAS
- simpleCUBLAS
- simpleCUBLAS_LU

# chatper 5 - domain specific

- p2pBandwidthLatencyTest

# chatper 6 - performance

- alignedTypes
- transpose
- UnifiedMemoryPerf
- LargeKernelParameter

# chatper 7
all examples are optional, they are quite interesting
