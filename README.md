# Metabolic traits analysis
Code to support analyses depending on metabolic traits.


## Conda environment
We are using `conda-lock` to ensure a reproducible environment. Please update the conda-lock files when updating the environment.

To create an environment from the lock file
```bash
conda create -n metabolic --file environment/conda-linux-64.lock
```
or
```bash
conda create -n metabolic --file environment/conda-osx-64.lock
```

## References

Deutsch, C., Penn, J. L., & Seibel, B. (2020). Metabolic trait diversity shapes marine biogeography. Nature, 585(7826), 557–562. https://doi.org/10.1038/s41586-020-2721-y
