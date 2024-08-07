# AggregationMethods

This repository contains all the source codes of consensual aggregation methods implemented in `R` and `python`. All codes contained in each file are built with recent version of `R` software (version $>$ 4.1, available [here](https://cran.r-project.org/bin/windows/base/)) and `Rstudio` (version > `2022.02.2+485`, available [here](https://www.rstudio.com/products/rstudio/download/#download)).

## Alternative

The aggregation methods implemented here are now available in `gradientcobra` Python library. The library can be installed by:

    `pip install gradientcobra`

For more information, see the library: [https://pypi.org/project/gradientcobra/](https://pypi.org/project/gradientcobra/).

---

## &#128270; How to download & run the codes in R?

To run the codes, you can <span style="color: #097BC1">`clone`</span> the repository directly or simply load the <span style="color: #097BC1">`R script`</span> source files from this repository using [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package in `Rstudio` as follows:

1. Install [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package using command: 

    `install.packages("devtools")`

2. Loading the source codes from `GitHub` repository using `source_url` function by: 

    `devtools::source_url("https://raw.githubusercontent.com/hassothea/AggregationMethods/main/file.R")`

where `file.R` is the file name contained in this repository which you want to import into your `Rstudio`.

---

## &#128270; How to download & run the codes in Python?

- Download and import the `.py` file into your environment.

## &#128214; Documentation

The documentation and explanation of all the aggregation methods are available on my webpage as listed below:

- `GradientCOBRARegressor` : for regression, see [GradientCOBRARegressor documentation](https://hassothea.github.io/files/CodesPhD/KernelAggReg.html).
- `KernelAggClassifier` : for classification, see [KernelAggClassifier documentation](https://hassothea.github.io/files/CodesPhD/KernelAggClass.html).
- `MixCobraRegressor` : for regression, see [MixCobraRegressor documentation](https://hassothea.github.io/files/CodesPhD/MixCobraReg.html).
- `MixCobraClassifier` : for classification, see [MixCobraClassifier documentation](https://hassothea.github.io/files/CodesPhD/MixCobraClass.html).

---
