# A Wrapper for methods of combining multiple tests

A custom implementation of all state of art methods for combining multiple tests, written from scratch in Python 3, API inspired by SciKit-learn. 
This module implements 8 methods:

* [SULIU](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476417)
* [LOGISTIC](https://academic.oup.com/biomet/article-abstract/54/1-2/167/331528?redirectedFrom=PDF)
* [STEPWISE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180010/)
* [MIN_MAX](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.4238?casa_token=Pumu2QRlkAgAAAAA%3AGb2f-XbvH7doJzpYZV35xxl7m1Ys_g9XENZKQlcB6z7-lV8b_7pN2uV-3XIUTqZu1Cl9fnzug2pL1OM)
* [RANDOM FOREST](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
* [SVMl](https://www.cambridge.org/core/books/an-introduction-to-support-vector-machines-and-other-kernelbased-learning-methods/A6A6F4084056A4B23F88648DDBFDD6FC)
* [SVMr](https://www.cambridge.org/core/books/an-introduction-to-support-vector-machines-and-other-kernelbased-learning-methods/A6A6F4084056A4B23F88648DDBFDD6FC)
* [PEPE](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1541-0420.2005.00420.x?casa_token=dwAgLvAvM5UAAAAA%3AVh00VmFM1QDWQGKbhqGWpfOG-x2oaHrzWYX8AbcB72VC3_h0V6C2N2Kg19x9W1yohh-Yv06XL062ZBs)

## Features

R and Python friendly

## Usage examples

Python
```javascript
from numpy.random import multivariate_normal
from utility.py import *. # download "utility.py" in the working directory.
u0 = [0.1,0.1,0.1, 0.1]; u1 = [0.6, 0.8, 1, 1.2]
sigma = [[1,0.5,0.5,0.5],
          [0.5,1,0.5,0.5],
          [0.5,0.5,1,0.5],
          [0.5,0.5,0.5,1]]
X0 = multivariate_normal(u0, sigma, size = 100)
X1 = multivariate_normal(u1, sigma, size = 80)
model = AllMethod(method= 'SULIU', bool_trans= False).fit(X0_train,X1_train)
_,_, auc = model.predict(X0_val,X1_val)
model.roc_plot(X0,X1)
```

R
```javascript
library(reticulate)
source_python("utility.py") ## download "utility.py" in the working directory.
mod = AllMethod(method= 'logistic', bool_trans = TRUE);
mod$fit(X0_train,X1_train); 
mod$predict(X0_val,X1_val)
mod$roc_plot(X0_val,X1_val)
```

## Built With

* [Dropwizard](https://scikit-learn.org/stable/modules/classes.html) - scikit-learn API
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Chengning Zhang** - *Initial work* - [Statistical methods for combining multiple diagnostic tests: a review](https://github.com/chengning-zhang/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details




