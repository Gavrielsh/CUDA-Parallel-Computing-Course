# Parallel Computing & CUDA Programming 🚀

This repository contains my coursework and projects for the **High-Performance Computing (HPC)** course. 
The projects demonstrate various parallel programming techniques using **C++** and **NVIDIA CUDA**.

## 🛠 Technologies & Tools
* **Languages:** C++, CUDA C
* **Hardware:** NVIDIA GPUs (RTX 3070, Tesla V100)
* **Concepts:** Parallel Algorithms, Memory Management (Global/Shared), Thread Divergence, Optimization, Amdahl's Law.
* **Environment:** Linux HPC Cluster, SLURM Workload Manager.

## 📂 Project List

### [Exercise 1: Numerical Integration (Riemann Sum)](./Ex1)
* **Description:** Implemented a parallel algorithm to approximate integrals using the Rectangle Rule.
* **Key Concepts:** Grid/Block structure, basic memory transfer (Host-to-Device), performance benchmarking vs CPU.
* **Results:** Achieved ~75x speedup on large datasets (N=100,000).
