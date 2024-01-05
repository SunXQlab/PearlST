# PearlST

## What is PearlST?

![PearlST Workflow](https://github.com/SunXQlab/PearlST/blob/main/PearlST-workflow.png)

PearlST (partial differential equation (PDE)-enhanced adversarial graph autoencoder of ST) is a tool that can precisely dissect spatial-temporal structures,   
including spatial domains, temporal trajectories, and signaling networks, from the spatial transcriptomics data.    
To this end, PearlST learns low-dimensional latent embeddings of ST by integrating spatial information,   
gene expression profiles and histology image features by leveraging PDE model-based gene expression augmentation and adversarial learning.   
The effectiveness of PearlST is extensively evaluated across multiple ST datasets obtained from various platforms, including 10X Visium, Stereo-seq, Slide-seqV2, MERFISH, and STARmap.

## How to access?

PearlST is an easily accessible and easy to use python package.  
Simply download the PearlST package from GitHub and unzip it onto your own computer to use PearlST's features to test your own datasets.

## How to use?

First, you need to make sure you have the dependency packages inside the requirements.txt file installed on your computer.   
Second, you need to change the parameters you want to tweak inside utilities.py.   
Third, run run_PearlST.py and you will get the corresponding results.

## Contact 
Please contact sunxq6@mail.sysu.edu.cn if you encounter any problems during use.
