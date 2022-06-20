# AggregationMethods

This repository contains all the source codes of consensual aggregation methods implemented in `R` programming. All codes contained in each file are built with recent version of `R` software (version $>$ 4.1, available [here](https://cran.r-project.org/bin/windows/base/)) and `Rstudio` (version > `2022.02.2+485`, available [here](https://www.rstudio.com/products/rstudio/download/#download)).

## &#128270; How to download & run the codes?

To run the codes, you can <span style="color: #097BC1">`clone`</span> the repository directly or simply load the <span style="color: #097BC1">`R script`</span> source files from this repository using [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package in `Rstudio` as follows:

1. Install [devtools](https://cran.r-project.org/web/packages/devtools/index.html) package using command: 

    `install.packages("devtools")`

2. Loading the source codes from `GitHub` repository using `source_url` function by: 

    `devtools::source_url("https://raw.githubusercontent.com/hassothea/AggregationMethods/main/file.R")`

where `file.R` is the file name contained in this repository which you want to import into your `Rstudio`.

## &#128214; Documentation

The documentation and explanation of all the aggregation methods are available on my webpage as listed below:

- `KernelAggReg` : for regression, see [KernelAggReg documentation](https://hassothea.github.io/files/CodesPhD/KernelAggReg.html).
- `KernelAggClass` : for classification, see [KernelAggClass documentation](https://hassothea.github.io/files/CodesPhD/KernelAggClass.html).
- `MixCobraReg` : for regression, see [MixCobraReg documentation](https://hassothea.github.io/files/CodesPhD/MixCobraReg.html).
- `MixCobraClass` : for classification, see [MixCobraClass documentation](https://hassothea.github.io/files/CodesPhD/MixCobraClass.html).
