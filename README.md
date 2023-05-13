![](https://img.shields.io/badge/language-python-yellow.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v1.0.0-519dd9.svg)
![](https://progress-bar.dev/20)

# 🧬 SIRS genomic analysis 🧬

[Systemic inflammatory response syndrome](https://en.wikipedia.org/wiki/Systemic_inflammatory_response_syndrome)
(SIRS) is an inflammatory state affecting the whole body. It is the body's response to an infectious or noninfectious insult.
Although the definition of SIRS refers to it as an "inflammatory" response, it actually has pro- and anti-inflammatory components.

The original studies examined pediatric patients admitted to the ICU, who were later classified as either SIRS (non-infectious) or Sepsis or Septic Shock (infectious).
There is also a group of healthy controls. Although the original studies examine patients at both ICU day 1 and ICU day 3, here we have only aggregated ICU day 1 patients.
All samples here were downloaded as .CEL files and re-normalized together using gcRMA using R package 'affy'.
The normalized data can be found on the series record and contains the gcRMA normalized expression values.

The dataset is composed of the unique patients (276; at the Day 1 timepoint) that are present in the six other GEO datasets published by Hector Wong and the Genomics of
Pediatric SIRS and Septic Shock Investigators. This dataset thus includes all unique patients from GSE4607, GSE8121, GSE9692, GSE13904, GSE26378, and GSE26440.
These are only from the Day 1 timepoint.

Initital data preprocessing was done by [1], differential gene expression (DGE) and re-normalization was done by [2]. 
I downloaded the data concerning SIRS from [bitbucket repository](https://bitbucket.org/i2rlab/rdea/src/master/) provided by [2]. 
The original data comes from [1] and is available via [Synapse](https://www.synapse.org/) as well as from 
[GEO GSE66099 dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66099). Part of the above description
was also taken from the same GEO dataset.

## 🔍 Inspect online

Because github does not support rendering `plotly` graphs, if you wish to inspect this project's notebooks online to their full extent,
please consider viewing this project tree on [NBviewer](https://nbviewer.org/github/f1lem0n/SIRS-genomics/tree/main/).
If you would like to explore the data through *interactive* plots that I have created that is the easiest way to go.

## 🐍 Environment

1. `git clone https://github.com/f1lem0n/SIRS-genomics` to your local machine
1. use any `conda` distribution to create and load the environment from the provided `conda-env.yml` file:
```bash
# inside the repo root directory
conda env create -f conda-env.yml -y
conda activate SIRS-genomics
```

## 📖 References

1. Timothy E Sweeney, Thanneer M Perumal, Ricardo Henao, Marshall Nichols, Judith A Howrylak, Augustine M Choi, Jesús F Bermejo-Martin, Raquel Almansa, Eduardo Tamayo, Emma E Davenport, et al. A community approach to mortality prediction in sepsis via gene expression analysis. Nature communications, 9(1):1–10, 2018.
1. Abbas, M., & EL-Manzalawy, Y. (2020). Machine learning based refined differential gene expression analysis of pediatric sepsis. In BMC Medical Genomics (Vol. 13, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1186/s12920-020-00771-4
