# Z score calculation & SE
# -----------------------------------------------------------------------------
import numpy as np
import scipy.stats as st
import math

def zscore(x, mean, stdev):
    return (x - mean) / stdev

z = zscore(135, 100, 15)
p = st.norm.cdf(z)
p_complement = 1 - p

SE = .07 / math.sqrt(29)



# Independent-samples t-test
# -----------------------------------------------------------------------------
t_lower = st.t.ppf(.025, 20)
t_upper = st.t.ppf(.975, 20)

print(t_lower, t_upper)

group1 = [211, 210, 210, 203, 196, 190, 191, 177, 173, 170, 163]
group2 = [181, 172, 196, 191, 167, 161, 178, 160, 149,  119, 156]
var1, var2 = np.var(group1), np.var(group2)

t_score, p_val = st.ttest_ind(group1, group2, equal_var=False)

print(t_score, p_val)



# F distribution for ANOVA:
# -----------------------------------------------------------------------------
from scipy.stats import f

dfn, dfd = 2, 202
fval = 5.69

f1 = f(dfn=dfn, dfd=dfd, loc=0, scale=1)
p_val = f1.pdf(fval)



# Chi square using contingency table
# -----------------------------------------------------------------------------
from scipy.stats import chi2_contingency
import numpy as np

observed_table = np.array([[5, 744], [9, 1489]])
chi2, p, dof, expectedTable = chi2_contingency(observed_table)



# McNemar test for paired case-control study using contingency table
# -----------------------------------------------------------------------------
from statsmodels.sandbox.stats.runs import mcnemar

contingencyTable = np.array([[13, 25], [4, 92]])
statistic, p = mcnemar(contingencyTable)



# kappa test for inter-observer agreement testing using contingency table
# -----------------------------------------------------------------------------
import numpy as np

def kappa_table(table):
    row_marg = np.sum(table, 0)
    col_marg = np.sum(table, 1)
    total = sum(row_marg)

    n = table.shape[1]
    diagonal_sum = sum(table[i][i] for i in range(n))
    observed = diagonal_sum / total

    expected = sum(row_marg * col_marg) / total**2

    kappa = (observed - expected) / (1 - expected)
    return kappa

observed_table = np.array([[123, 10], [6, 29]])
k = kappa_table(observed_table)
