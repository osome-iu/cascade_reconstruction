# Information diffusion assumptions can distort our understanding of social network dynamics

This repository stores code for a research project that explores the effect of reconstructing Bluesky and Twitter cascades on downstream analyses.
It also explores the impact of different reconstruction assumptions.
For this study, we introduce a novel reconstruction method called Probablistic Diffusion Inference.
Please see the manuscript, linked below, for all details.

## Paper

For more details, you can find the paper [here]() (TO BE INCLUDED). It should be cited as:

[Matthew R. DeVerna](https://www.matthewdeverna.com/), [Francesco Pierri](https://pierri.faculty.polimi.it/), [Rachith Aiyappa](https://rachithaiyappa.github.io/), [Diogo Pachecho](https://diogofpacheco.github.io/), [John Bryden](https://jbryden.co.uk/home/), and [Filippo Menczer](https://cnets.indiana.edu/fil) (2024). **Information diffusion assumptions can distort our understanding of social network dynamics**, ArXiv preprint:TO BE INCLUDED. doi: ADD ARXIV LINK HERE.


```bib
@article{deverna2023cascades,
  title={Information diffusion assumptions can distort our understanding of social network dynamics},
  author={DeVerna, Matthew R. and Pierri, Francesco and Aiyappa, Rachith Pachecho, Diogo and Bryden, John and Menczer, Filippo},
  journal={Preprint arXiv:UPDATE ME},
  year={2024}
}
```


### Contents
- `bluesky/`: bluesky related code saved here.
- `cleaned_data/`: cleaned Vosoughi et al data (from [this paper](https://doi.org/10.1126/science.aap9559)) is saved here.
- `code/`: code that analyzes the Vosoughi data is saved here.
- `environments/`: virtual environment files are saved here.
- `generate_figures/`: code to generate all figures for the paper is saved here.
- `generate_statistics/`: code to generate all statistics reported within the paper is saved here.
- `info_diff_review/`: code to collect and clean OpenAlex data is here.
- `midterm/`: midterm/twitter related code is saved here.
- `output/`: output files from analyses of the Vosoughi data are saved here.
- `vosoughi_replication_code/`: Vosoughi's replication code and data.



## Replication notes

> **Important notes!!**: 
> 1. Before replicating this analysis be sure that you have the required computational resources.
> **This code will require that you generate many millions of files and can crash your system** by exhausting your machine's "inode" resources (see [this](https://en.wikipedia.org/wiki/Inode#Potential_for_inode_exhaustion_and_solutions)).
> This happens because the code generates an individual network file for every version of a cascade that we generate.
> E.g., in the Vosoughi dataset, we reconstruct over 40k cascades. We generate 100 versions of each and we do this for nine different parameter settings. The rough math on the number of files here is 40,000 * 100 * 9 = 36,000,000 files. We also do the same thing for the Bluesky and Twitter data so the number of files is much larger.
> 2. Given these computational requirements, (1) many scripts write somewhere within the directory `/data_volume/cascade_reconstruction/`.
> The contents of this directory can be found within the `cascade_reconstruction.tar.gz` tarball, which you can decompress via `tar -xzvf cascade_reconstruction.tar.gz`.
> After decompressing this archive, there will be additional tarballs within the subfolders that will need to be found and further decompressed.
> Finally, before attempting to replicate, make sure that paths within scripts match. The path to this directory should be `/data_volume/cascade_reconstruction/`.

### Requirements

Python: See the `environments/` directory for details on installing the correct virtual environment for replication.


### Generating figures and statistics

After downloading the Zenodo data and activating the virtual environment, you can generate all figures and statistics from the paper by executing the below command.

```sh
bash generate_figures_and_statistics.sh
```


### Rerunning the entire analysis pipeline


#### Step 1: Install the local packages

There are two local packages to install. Go to where they are saved...

- `code/package/`
- `midterm/code/package/`

... and follow the instructions in the `README.md` files to install them into the project's virtual environment.

#### Step 2: Run the pipeline

You can run the entire analysis pipeline with the below command.

```sh
bash run_pipeline.sh
```

### Questions

All questions can be directed to [Matt DeVerna](https://www.matthewdeverna.com/) or listed as an issue on the public [GitHub repository](https://github.com/osome-iu/cascade_reconstruction).