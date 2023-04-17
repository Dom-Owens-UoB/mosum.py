# mosum.py: Moving Sum Based Procedures for Changes in the Mean

A python port of the R package mosum <https://CRAN.R-project.org/package=mosum>.
Implementations of MOSUM-based statistical procedures and algorithms for detecting multiple changes in the mean. 
This comprises the MOSUM procedure for estimating multiple mean changes from Eichinger and Kirch (2018) <doi:10.3150/16-BEJ887> 
and the multiscale algorithmic extension from Cho and Kirch (2022) <doi:10.1007/s10463-021-00811-5>, 
as well as the bootstrap procedure for generating confidence intervals about the locations of change points as proposed in Cho and Kirch (2022) <doi:10.1016/j.csda.2022.107552>. 
See also Meier, Kirch and Cho (2021) <doi:10.18637/jss.v097.i08> which accompanies the R package.

## Installation

```bash
$ pip install mosum
```

## Quick start

`mosum.py` can be used as follows to detect changes in the mean of a time series

```python
import mosum
#   simulate data
xx = mosum.testData("blocks")["x"]
# detect changes
xx_m  = mosum.mosum(xx, G = 50, criterion = "eta", boundary_extension = True)
# summary and print methods
xx_m.summary()
xx_m.print()
# plot the output
xx_m.plot(display="mosum")
from matplotlib import pyplot as plt
plt.show()
```

## Usage 

See [usage](https://dom-owens-uob.github.io/mosum.py/docs/usage.html)
for a detailed description of how to use the package.
## License

mosum.py was created by Dom Owens, based on the R package "mosum", originally by Alexander Meier, Haeran Cho, and Claudia Kirch.
It is licensed under the terms of the MIT license.