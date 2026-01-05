=============================================================================
HW1 - Introduction to CUDA: Numerical Integration
=============================================================================

Student Name: Gavriel Shalem
Student ID:   205461486

-----------------------------------------------------------------------------
1. GENERAL DESCRIPTION
-----------------------------------------------------------------------------
This program calculates the approximate area under a curve (integral) using 
the Riemann Sum method (Rectangle Rule).
The solution compares a sequential CPU implementation against a parallel 
GPU implementation using CUDA.

-----------------------------------------------------------------------------
2. IMPLEMENTATION DETAILS (MAPPING TO REQUIREMENTS)
-----------------------------------------------------------------------------

A. Integration Logic (The Kernel)
   - The parallel calculation is implemented in the global function:
     'my_kernel' (lines 60-77).
   - Each thread calculates the area of one specific rectangle and writes 
     it to a global array.
   - The final summation is performed on the Host after copying results back.

B. Verification (Polynomial Function)
   - Requirement: Verify correctness using f(x) = 10x^3 + 2x^2 - 7x + 6.
   - Implementation:
     * Device function: 'f_poly'
     * Host function: 'cpu_poly'
     * Execution: 'Part A' inside the main function.
   - We compare the GPU result against the exact mathematical integral
     implemented in 'get_exact'.

C. Benchmarking (Heavy Function)
   - Requirement: Test performance with f(x) = sum(sin(...) * cos(...)).
   - Implementation:
     * Device function: 'f_heavy' (using a while loop to simulate load).
     * Host function: 'cpu_heavy'.
     * Execution: 'Part B' inside the main function.
   - The program loops through N = 100, 1000, 10000, 100000.

D. Timing & Speedup
   - GPU Time: Measured using 'cudaEventRecord' (t1, t2) to exclude 
     memory allocation overhead, but include kernel execution.
   - CPU Time: Measured using 'clock()'.
   - Speedup: Calculated as (CPU_Time / GPU_Time).

-----------------------------------------------------------------------------
3. FILES IN THE SUBMISSION
-----------------------------------------------------------------------------
1. ex1.cpp      - Source code (Host + Device).
2. ex1.sbatch   - Script for running on the cluster (SLURM).
3. README.txt   - This file.

-----------------------------------------------------------------------------
4. HOW TO RUN
-----------------------------------------------------------------------------
Option 1: Using SLURM (Recommended)
   Run the command: 
   sbatch ex1.sbatch
   
   (The output will be saved to out_<job_id>.txt)

Option 2: Manual Compilation (Interactive)
   If running interactively, verify paths and compile:
   
   export PATH=/usr/local/cuda/bin:$PATH
   nvcc -x cu ex1.cpp -o my_prog -O3
   ./my_prog

-----------------------------------------------------------------------------
5. RESULTS SUMMARY
-----------------------------------------------------------------------------

A. Verification (Polynomial Function)
   - Exact Result: 6411930.000000
   - GPU Result:   6410980.848840
   - Error:        0.0148 %
   
   Conclusion: The logic is correct. The minor error is due to the Riemann 
   Sum approximation method and floating-point precision, not a code bug.

B. Benchmarking (Heavy Function)
   Here are the measurements from the cluster execution:

   N         CPU Time (s)    GPU Time (s)    Speedup
   --------------------------------------------------------
   100       0.0639          0.0493          1.30 x
   1000      0.5943          0.0929          6.40 x
   10000     5.9390          0.0937          63.39 x
   100000    59.6347         0.7859          75.88 x

C. Analysis of Results
   1. Small N (100): The speedup is negligible (1.3x). At this scale, the 
      overhead of copying memory (Host->Device) and launching the kernel 
      takes more time than the actual calculation.
   
   2. Medium N (1,000 vs 10,000): 
      Interestingly, the GPU time is almost identical for N=1,000 (0.0929s) 
      and N=10,000 (0.0937s), while the CPU time jumped by 10x. 
      This proves that the GPU has enough cores to handle 10x more work 
      in parallel without taking extra time.

   3. Large N (100,000): The CPU struggled significantly (~1 minute), while 
      the GPU finished in under a second. The max speedup of ~76x 
      demonstrates the massive power of parallel computing for independent 
      mathematical tasks.