# CausalKit

## A Package for Causal Inference inspired by econometrics and modern ML API design

CausalKit is a Python package designed for students and researchers alike. It offers a simple approach to economic and statistical analysis, emphasizing ease of use, interpretability and causation. 

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

Upcoming Features to implement:

1. add dropping of na values automatically upon model instantiation
2. add IV Regression use angrist data as example
3. add t test
4. add random effect and mixed effects models
5. add model diagnostics + assumption checks (linearity, normality of residuals, homoscedasticity, and absence of multicollinearity. This could include plots (like QQ plots, residual vs. fitted value plots) and statistical tests.)
6. Regularization (Lasso, Ridge, Elastic Net)
7. add GLM's (poisson, negative binomial, multinomial)
8. Bootstrap + resampling methods (bootstrap se ci, permutation tests)
9. add typical Causal inferenece methods interfaces (matching, regression discontinuity, synthetic control, etc.)
10. interactive interaction explorer (interactive visualization with sliders for continuous variables and dropdowns for categorical variables)
11. Refactor Fixed Effects class to handle high dimensional fixed effects efficiently, add method to support within estimator
