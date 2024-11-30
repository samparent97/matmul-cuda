[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/EFHW-UVA)

# CE 4SP4 - Lab4

## Description
In Lab 4, you will implement parallel matrix-matrix multiplication on GPU.

## Objectives
- Parallel programming in GPU using OpenCL

## Some notes
* You will need to first finish the TODO items in the code and then change your implementations. Please remove TODOs 
once implement them.
 to answer following questions in the report (see questions in the following parts). Your implementations should work correctly for ANY cornel cases.
* Please push a PDF of your report in the `doc` folder.
The `README` file in the `doc` folder provides some hints how to prepare a good report. 
* The evaluation is based on the quality of the report and 
the correctness/efficiency of the implementation. So make sure to provide all the necessary details in the report.
The report should be single column with font size 11 or 12. The report should be at most 6 pages long, 
including figures, references, tables,.... . A good report should be clear and concise and backed up with data/plots.
* A good sample for plots and tables are those provided in lecture slides.
* All group members should contribute to all parts of the lab. If contributions are not equal, please specify at the 
end of the report. 

## Installation and Running the Code
The installation instructions for the teach cluster is provided as a bash script. You will need to clone 
the repository. Then you can emit the 
following commands to install the code on the teach cluster after cloning the repository:

```bash
cd 4sp4-lab04 folder
bash build.sh
```

To run the code on the teach cluster:
```bash
sbatch run_teach_cluster.sh mm
```


**Please do NOT edit SLURM part of the script.**

## Part 1: Matrix-Matrix Multiplication on GPU
- Implement three different task decompositions for dense matrix-matrix multiplication on GPU. You should have three GPU implementations:
  - single row-based decomposition
  - multiple-row-based decomposition (1D tiling on rows of A)
  - 1D tiling on rows of A + 1D tiling on rows or columns of B

- In your report, you will need to discuss:
    - Speedup plots over the CPU baseline for each implementation.
    - Comparing each decomposition (show a plot) and discuss their performance for square, tall skinny, and short wide matrices (use number of plots). 
  Your implementation does not need to be tailored for each case. You can use the same implementation for all cases. But you need to discuss the performance for each case.
    - The effect of different block size (work-group size) should be discussed and the selected work-group should be justified.
- Note: the ratios of number of rows to number of columns in tall skinny matrices and short wide matrices are 100 and 0.01, respectively.