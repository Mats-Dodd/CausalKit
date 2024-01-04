# CausalKit

## A Package for Causal Inference Inspired by Econometrics

CausalKit is a Python package specifically designed for economics students and researchers. It offers a unique approach to regression analysis, emphasizing interpretability and causal inference, all through an econometrics lens.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Test Coverage](https://img.shields.io/badge/coverage-100-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.89+-blue.svg)
### Features

- **Econometrics-Driven Approach:** Our methods are rooted in econometrics principles, ensuring robust and reliable analysis.
- **Focus on Causal Inference:** Understand the 'why' behind your data with tools designed for causal analysis.
- **Intuitive Interface:** Designed with simplicity in mind, making it accessible for students and professionals alike.
- **Comprehensive Documentation:** Detailed guides and examples to help you get started and make the most out of CausalKit.

### Installation

```bash
pip install causalkit
```

Things to implement:

1. add dropping of na values
4. add IV Regression use angrist data as example
5. add t test
7. add random effect and mixed effects models
8. add model diagnostics + assumption checks (linearity, normality of residuals, homoscedasticity, and absence of multicollinearity. This could include plots (like QQ plots, residual vs. fitted value plots) and statistical tests.)
9. Regularization (Lasso, Ridge, Elastic Net)
10. add GLM's (logistic, poisson, negative binomial, multinomial)
11. Bootstrap + resampling methods (jackknife, permutation tests, )
12. add typical Causal inferenece methods (matching, regression discontinuity, difference in differences,  regression kink, synthetic control, etc.)
13. interactive interaction explorer (interactive vis with sliders for continuous variables and dropdowns for categorical variables)
