Steps to Run in collab 
=========================================

Step 1: Install Dependencies
----------------------------
- Download and install all necessary dependencies, including the CUDA Toolkit.

Step 2: Preparing Data (For Google Colab)
-----------------------------------------
- Upload the Amazon dataset to your Colab environment.
- Run the provided Python code to convert the dataset into 'vertex.CSR' and 'edge.CSR' format.

Step 3: Run Compression Analysis
--------------------------------
- Run the EFG compression implementation and measure the compression ratio.

Step 4: Run SSSP using EFG Format
---------------------------------
- Compile and run the CUDA-based SSSP algorithm using the EFG format.

Step 5: Run SSSP using CSR Format
---------------------------------
- Execute the baseline SSSP implementation using CSR traversal.

Running on Local Host
=====================

Note: All CUDA files are located in the 'cuda_code' folder.

After installing the dependencies, use the following commands:

1. Run CSR Compression
----------------------
- This step calculates the compression ratio.

Command:
--------
nvcc csr_compression.cu -o csr_compression -arch=sm_70
./csr_compression

2. Run Baseline SSSP (CSR Format)
---------------------------------
- This is the standard SSSP algorithm using CSR.

Command:
--------
nvcc -o graph_ef_cuda graph_ef_cuda.cu -arch=sm_70 -lcudart
./graph_ef_cuda

3. Run SSSP with EFG Format
---------------------------
- This runs the SSSP implementation with EFG compression.

Commands:
---------
nvcc -o efg_cuda efg_cuda.cu
ls -lh vertex.CSR edge.CSR
./efg_cuda
