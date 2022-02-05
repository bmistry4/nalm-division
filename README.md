# Learning Division with Neural Arithmetic Logic Modules
This repository is the official implementation of [Learning Division with Neural Arithmetic Logic Modules](TODO).

This work builds ontop of the research on **[Neural Arithmetic Units](https://openreview.net/forum?id=H1gNOeHKPS) 
by Andreas Madsen and Alexander Rosenberg Johansen**. 
The [original code](https://github.com/AndreasMadsen/stable-nalu) is by Andreas Madsen, who created the 
underlying framework used to create datasets, run experiments, and generate plots. 
**See their original README ([below](#neural-arithmetic-units))** (which includes requirements).

## About 
[Neural Arithmetic Logic Modules](https://arxiv.org/abs/2101.09530) are neural networks with the ability to learn arithmetic operations in a systematic
manner, or more simply put, a network where the weights are interpretable representing input selection and an specific 
operation. 
This work focuses specifically on learning division, where we evaluate an existing division module 
(the [**Real NPU**](https://arxiv.org/pdf/2006.01681v2.pdf])) and create two new modules the **Neural Reciprocal Unit** 
and the **Neural Multiplicative Reciprocal Unit** in the process to unravel:
- Why learning division is hard?
- What components of NALMs make learning division easier? 

## Recreating Experiments From the Paper: Training & Evaluation
### Single Module Task
First, create a csv file containing the threshold values for each range using 
<pre> Rscript <a href="export/single_layer_task/generate_exp_setups.r">generate_exp_setups.r</a> </pre>

#### Generating plots consists of 3 stages
1. Run a shell script which calls the python script to _generate the tensorboard results_ over multiple seeds and ranges
    - `sbatch lfs_batch_jobs/single_layer_task/neurips_2021/<script name>.sh`
        - `<script name>`: Refer to the table below.
2. Call the python script to convert the tensorboard _results to a csv file_
    - `python3 export/simple_function_static.py --tensorboard-dir 
/data/nalms/tensorboard/<experiment_name>/ --csv-out /data/nalms/csvs/<experiment_name>.csv`
        - `--tensorboard-dir`: Directory containing the tensorboard folders with the model results
        - `--csv-out`: Filepath on where to save the csv result file
        - `<experiment_name>`: value of the experiment_name variable in the shell script used for step 1

3. Call the R script to convert the csv results to a _plot_ (saved as pdf)
    - <pre> Rscript <a href="export/single_layer_task/neurips_range.r">neurips_range.r</a> None /data/nalms/csvs/r_results/neurips-2021/ sltr-in2 op-div None nips-sltr-in2 </pre>
        - First arg: N/A
        - Second arg: Path to directory where you want to save the plot file
        - Third arg: Filename for plot(/ loading csv filename if single model). Use lookup key value (see table below).
        - Forth arg: Arithmetic operation to create plot of (i.e. op-add, op-sub, op-mul, and op-div)
        - Fifth arg: N/A
        - Sixth arg: Lookup key (see table below) used to load relevant files and plot information
    

##### Experiment Meta-Information Table
| Figure | Experiment                      | Shell script name                                                                                        | Lookup key              |
|--------|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------|
| 1a     | L1 regularisation               | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)                             | nips-realnpu-L1             |
| 1b     | L1 beta sweep                   | [in2-realnpu-beta-sweep](lfs_batch_jobs/single_layer_task/neurips_2021/in2-realnpu-beta-sweep.sh)    | nips-realnpu-L1_sweep       |
| 2a     | Clipping                        | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)                             | nips-realnpu-clipping       |
| 2b     | Discretisation                  | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)                             | nips-realnpu-discretisation |
| 2c     | Initalisation                   | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)                             | nips-realnpu-init           |
| 3      | No redundancy (input size 2)    | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)                             | nips-sltr-in2               |
| 4      | Mixed-signed inputs             | [in2-mixed-signs.sh](lfs_batch_jobs/single_layer_task/neurips_2021/in2-mixed-signs.sh)               | N/A (see section below)     |
| 6      | With redundancy (input size 10) | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh)                           | nips-sltr-in10              |


#### Mixed-signed Inputs (on the modified Real NPU) 
Generate the tensorboard results and the csv file using the first two stages. 

To generate the plot, run:  
<pre> Rscript <a href="export/single_layer_task/neurips_range.r">neurips_range.r</a> /data/nalms/csvs/sltr-in2/ /data/nalms/csvs/r_results/neurips-2021/ mixed-sign-ds_realnpu-modified op-div None</pre>

### Division by Small Numbers (Figure 5)
This does not require running the 3 stages. Instead:
1. Generate gold test error csv: `python3 export/single_layer_task/generate_divBy0_extrap_thresholds.py`
2. Create plot: <pre> Rscript <a href="export/single_layer_task/divBy0_gold_test_errors.r">divBy0_gold_test_errors.r</a></pre>

### More Challenging Distributions (Figure 6 and 7)
Generate the tensorboard results and the csv file using the first two stages. 

To generate the Figure 6, run:  
<pre> Rscript <a href="export/single_layer_task/neurips_range_distributions.r">neurips_range_distributions.r</a> /data/nalms/csvs/sltr-in2/distributions/ /data/nalms/csvs/r_results/neurips-2021/ sltr-in2-distributions op-div None nips-in2-distributions</pre>

To generate the Figure 7, run:  
<pre> Rscript <a href="export/single_layer_task/neurips_range_distributions.r">neurips_range_distributions.r</a> /data/nalms/csvs/sltr-in10/distributions/ /data/nalms/csvs/r_results/neurips-2021/ sltr-in10-distributions op-div None nips-in10-distributions</pre>


### RMSE Loss Landscape (Figure 9)
See Collab [Notebook](notebook/2_layer_singularity_issue.ipynb) 

## Appendix
Use the same [3 stages](https://github.com/bmistry4/nalu-stable-exp/blob/master/README.md#generating-plots-consists-of-3-stages) to generate plots. 

| Figure | Experiment                      | Shell script name                                                          | Lookup key                      |
|--------|---------------------------------|----------------------------------------------------------------------------|---------------------------------|
| 10     | NRU - Learning rates            | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)   | nips-in2-nru-lr                 |
| 11     | DivBy0 - [a] to 1/a             | [divBy0.sh](lfs_batch_jobs/single_layer_task/neurips_2021/divBy0.sh)       | -                               |
| 12     | DivBy0 - [a,b] to 1/a           | [divBy0.sh](lfs_batch_jobs/single_layer_task/neurips_2021/divBy0.sh)       | -                               |
| 13     | DivBy0 - [a,b] to a/b           | [divBy0.sh](lfs_batch_jobs/single_layer_task/neurips_2021/divBy0.sh)       | -                               |
| 14     | Real NPU - L2 regularisation    | [sltr-in2.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in2.sh)   | nips-realnpu-L2                 |
| 15     | NPU                             | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-sltr-in10-npu              |
| 16     | Real NPU - NAU discretisation   | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-in10-realnpu-W-reg         |
| 17     | NMRU - Ablation                 | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-in10-nmru-ablation         |
| 18     | NMRU - Learning rates           | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-in10-nmru-lr               |
| 19     | NMRU - Optimiser                | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-in10-nmru-optimiser        |
| 20     | NRU - Separate signs            | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-in10-nru-separate-mag-sign |
| 21     | Losses - Real NPU               | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-sltr-in10-losses-realnpu   |
| 22     | Losses - NRU                    | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-sltr-in10-losses-nru       |
| 23     | Losses - NMRU                   | [sltr-in10.sh](lfs_batch_jobs/single_layer_task/neurips_2021/sltr-in10.sh) | nips-sltr-in10-losses-nmru      |

Any experiments with different steps are explained below.

### Division by Small Numbers - Experimental Results (Appendix Figures 11-13)
1. Generate the extrapolation thresholds using `python3 export/single_layer_task/generate_divBy0_extrap_thresholds.py` 
with `eps=torch.finfo().eps`.
2. Copy thresholds into the relevant cells in [exp_setups.csv](export/single_layer_task/exp_setups.csv).
3. Generate the tensorboard results using `bash <script name> 0 24` 
4. Convert tensorboard to csv results (using the usual command) 
4. Run the following commands to generate the plots for each of the three tasks:
    - [a] to 1/a: 
    <pre> Rscript <a href="export/single_layer_task/neurips_range_divBy0.r">neurips_range_divBy0.r</a> /data/nalms/csvs/SLTR_divBy0/easy /data/nalms/csvs/r_results/neurips-2021/divBy0/ divBy0-easy op-reciprocal None nips-divBy0-easy zero.range.easy</pre>

    - [a,b] to 1/a:
    <pre> Rscript <a href="export/single_layer_task/neurips_range_divBy0.r">neurips_range_divBy0.r</a> /data/nalms/csvs/SLTR_divBy0/medium /data/nalms/csvs/r_results/neurips-2021/divBy0/ divBy0-medium op-reciprocal None nips-divBy0-medium zero.range.medium</pre>

    - [a,b] to a/b: 
    <pre> Rscript <a href="export/single_layer_task/neurips_range_divBy0.r">neurips_range_divBy0.r</a> /data/nalms/csvs/SLTR_divBy0/hard /data/nalms/csvs/r_results/neurips-2021/divBy0/ divBy0-hard op-div None nips-divBy0-hard zero.range.hard</pre>

### (Note - Scipy package version)
If you want to have samples from a truncated normal distribution then the scipy version installed must be
1.6.2 (older versions than 1.6 sample for this distribution too slowly). 
`pip install --upgrade scipy==1.6.2`

---
# Neural Arithmetic Units

This code encompass two publiations. The ICLR paper is still in review, please respect the double-blind review process.

![Hidden Size results](readme-image.png)

_Figure, shows performance of our proposed NMU model._

## Publications

#### SEDL Workshop at NeurIPS 2019

Reproduction study of the Neural Arithmetic Logic Unit (NALU). We propose an improved evaluation criterion of arithmetic tasks including a "converged at" and a "sparsity error" metric. Results will be presented at [SEDL|NeurIPS 2019](https://sites.google.com/view/sedl-neurips-2019/#h.p_vZ65rPBhIlB4). – [Read paper](http://arxiv.org/abs/1910.01888).

```bib
@inproceedings{maep-madsen-johansen-2019,
    author={Andreas Madsen and Alexander Rosenberg Johansen},
    title={Measuring Arithmetic Extrapolation Performance},
    booktitle={Science meets Engineering of Deep Learning at 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)},
    address={Vancouver, Canada},
    journal={CoRR},
    volume={abs/1910.01888},
    month={October},
    year={2019},
    url={http://arxiv.org/abs/1910.01888},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    eprint={1910.01888},
    timestamp={Fri, 4 Oct 2019 12:00:36 UTC}
}
```

#### ICLR 2020 (Under review)

Our main contribution, which includes a theoretical analysis of the optimization challenges with the NALU. Based on these difficulties we propose several improvements. **This is under double-blind peer-review, please respect our anonymity and reference https://openreview.net/forum?id=H1gNOeHKPS and not this repository!** – [Read paper](https://openreview.net/forum?id=H1gNOeHKPS).

```bib
@inproceedings{mnu-madsen-johansen-2020,
    author={Andreas Madsen and Alexander Rosenberg Johansen},
    title={Neural Arithmetic Units},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=H1gNOeHKPS},
    note={under review}
}
```

## Install

```bash
python3 setup.py develop
```

This will install this code under the name `stable-nalu`, and the following dependencies if missing: `numpy, tqdm, torch, scipy, pandas, tensorflow, torchvision, tensorboard, tensorboardX`.

## Experiments used in the paper

All experiments results shown in the paper can be exactly reproduced using fixed seeds. The `lfs_batch_jobs`
directory contains bash scripts for submitting jobs to an LFS queue. The `bsub` and its arguments, can be
replaced with `python3` or an equivalent command for another queue system.

The `export` directory contains python scripts for converting the tensorboard results into CSV files and
contains R scripts for presenting those results, as presented in the paper.

## Naming changes

As said earlier the naming convensions in the code are different from the paper. The following translations
can be used:

* Linear: `--layer-type linear`
* ReLU: `--layer-type ReLU`
* ReLU6: `--layer-type ReLU6`
* NAC-add: `--layer-type NAC`
* NAC-mul: `--layer-type NAC --nac-mul normal`
* NAC-sigma: `--layer-type PosNAC --nac-mul normal`
* NAC-nmu: `--layer-type ReRegualizedLinearPosNAC --nac-mul normal --first-layer ReRegualizedLinearNAC`
* NALU: `--layer-type NALU`
* NAU: `--layer-type ReRegualizedLinearNAC`
* NMU: `--layer-type ReRegualizedLinearNAC --nac-mul mnac`

## Extra experiments

Here are 4 experiments in total, they correspond to the experiments in the NALU paper.

```
python3 experiments/simple_function_static.py --help # 4.1 (static)
python3 experiments/sequential_mnist.py --help # 4.2
```

Example with using NMU on the multiplication problem:

```bash
python3 experiments/simple_function_static.py \
    --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    --seed 0 --max-iterations 5000000 --verbose \
    --name-prefix test --remove-existing-data
```

The `--verbose` logs network internal measures to the tensorboard. You can access the tensorboard with:

```
tensorboard --logdir tensorboard
```
