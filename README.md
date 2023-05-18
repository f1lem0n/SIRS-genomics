![](https://img.shields.io/badge/language-python-yellow.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v1.0.0-519dd9.svg)
![](https://progress-bar.dev/50)

# 🧬 SIRS morbidity prediction from expression profiles

[Systemic inflammatory response syndrome](https://en.wikipedia.org/wiki/Systemic_inflammatory_response_syndrome)
(SIRS) is an inflammatory state affecting the whole body. It is the body's response to an infectious or noninfectious insult.
Although the definition of SIRS refers to it as an "inflammatory" response, it actually has pro- and anti-inflammatory components.

This analysis was focused on finding a suitable model for morbidity prediction from gene expression profile of the patient.
Initital data preprocessing was done by [1], differential gene expression (DGE) analysis and re-normalization was done by [2].
I downloaded the data concerning SIRS from [bitbucket repository](https://bitbucket.org/i2rlab/rdea/src/master/) provided by [2].
The original data comes from [1] and is available via [Synapse](https://www.synapse.org/) (after logging in) and
[GEO GSE66099 dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66099).

## 🔍 Inspect online

Because github does not support rendering `plotly` graphs, if you wish to inspect this project's notebooks online to their full extent,
please consider viewing this project tree on [NBviewer](https://nbviewer.org/).
If you would like to explore the data through *interactive* plots that I have created that is the easiest way to go.

⚠️ If for some reason NBviewer renders an incomplete version of the notebook, try adding `?flush_cache=true` at the end of the URL.

## 🐍 Environment

1. `git clone https://github.com/f1lem0n/SIRS-genomics` to your local machine
2. use any `conda` distribution to create and load the environment from the provided `.yml` file:

```bash
# inside the repo root directory
conda env create -f conda-env-deb12.yml -y  # debian-based linux distros
conda env create -f conda-env-win10.yml -y  # windows
conda activate SIRS-genomics
```

## 📖 References

1. Timothy E Sweeney, Thanneer M Perumal, Ricardo Henao, Marshall Nichols, Judith A Howrylak, Augustine M Choi, Jesús F Bermejo-Martin, Raquel Almansa, Eduardo Tamayo, Emma E Davenport, et al. A community approach to mortality prediction in sepsis via gene expression analysis. Nature communications, 9(1):1–10, 2018.
2. Abbas, M., & EL-Manzalawy, Y. (2020). Machine learning based refined differential gene expression analysis of pediatric sepsis. In BMC Medical Genomics (Vol. 13, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1186/s12920-020-00771-4
