# hypotest
In this repository I summarize some of the most widely used hypothesis tests. <br>

The notebook ```hypotests_nanova.ipynb``` contains commonly used tests for one or two samples, including Z-test, t-test, Wilcoxon signed rank test, rank-sum test and permutation test. <br>

The notebook ```hypotests_anova.ipynb``` contains commonly used tests for comparing difference in mean of more than 2 groups, including one-way F test (simple one-way ANOVA), Kruskal-Wallis test (the non-parametric alternative to simple one-way anova), r-ANOVA  for one way correlated samples, Friedman test (the non-parametric alternative to rANOVA) and Tukey HSD test (one post-hoc analysis for one-way ANOVA). <br>

The notebook ```hypotests_equal_variance.ipynb``` contains commonly used tests of homogeneity of variance among samples. Tests of one sample variance equal to a constant include Chi-square test for variance and Wald test with bootstrap to estimate standard error of sample variance. Tests of equality of variance among different samples include F-test, Hartley's test, Bartlett's test, Levene's test and permutation test.

In each notebook, I explain the assumptions, brief derivations and applicatibity of each test followed by examples using fake data created using python.
