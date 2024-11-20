# Math Reference

## Log Transform

Using the $ \log $ transform (14) of the salt differences, try to establish a correlation.

#### Log Transform Derivation

**Assume all local inflows have the same salinity, and all outflows have the salinity at the downstream gauge.**

Given by the water mass balance model, we have:

1) $$flow_{in} + flow_{local} = flow_{out}$$

Assume that all terms in $ flow_{out} $ are the same salinity $ sal_{out} $, define the salt mass balance:

2) $$ flow_{out} (sal_{out}) = flow_{in} (sal_{in}) + flow_{local} (sal_{local}) $$

Let the relative salinity concentrations at $ sal_{out} $ and $ sal_{local} $ be defined as:

3) $$ \Delta sal_{out} = sal_{out} - sal_{in} $$

4) $$ \Delta sal_{local} = sal_{local} - sal_{in} $$

Substituting (3) leads to:

5) $$ flow_{out} (sal_{out}) = flow_{out} (sal_{in} + \Delta sal_{out}) $$

Substituting (4) leads to:

6) $$ flow_{out} (sal_{out}) = flow_{in} (sal_{in}) + flow_{local} (sal_{in} + \Delta sal_{local}) $$

Taking (5) and (6) leads to:

7) $$ flow_{out} (sal_{in} + \Delta sal_{out}) = flow_{in} (sal_{in}) + flow_{local} (sal_{in} + \Delta sal_{local}) $$

Substituting for $ flow_{out} $ from (1):

8) $$ (flow_{in} + flow_{local})(sal_{in} + \Delta sal_{out}) = flow_{in} (sal_{in}) + flow_{local} (sal_{in} + \Delta sal_{local}) $$

Expand both sides:

9) $$ flow_{in} (sal_{in}) + flow_{in} (\Delta sal_{out}) + flow_{local} (sal_{in}) + flow_{local} (\Delta sal_{out}) = flow_{in} (sal_{in}) + flow_{local} (sal_{in}) + flow_{local} (\Delta sal_{local}) $$

Eliminate like terms:

10) $$ flow_{in} (\Delta sal_{out}) + flow_{local} (\Delta sal_{out}) = flow_{local} (\Delta sal_{local}) $$

Substitute $ flow_{out} $ from (1):

11) $$ flow_{out} ( \Delta sal_{out} ) = flow_{local} (\Delta sal_{local}) $$

Which can be rewritten as:

12) $$ \Delta sal_{out} = \frac{flow_{local} (\Delta sal_{local})}{flow_{out}} $$

Take the $ \log $ of both sides:

13) $$ \log ( \Delta sal_{out} ) = \log ( \frac{flow_{local} (\Delta sal_{local})}{flow_{out}} ) $$

Expand the terms:

14) $$ \log ( \Delta sal_{out} ) = \log ( \Delta sal_{local} ) + \log ( flow_{local} ) - \log ( flow_{out} )$$

Eq. (12) can also be written as:

15) $$ \Delta sal_{out} = sal_{local}(\frac{flow_{local}}{flow_{out}}) - sal_{in}(\frac{flow_{local}}{flow_{out}})$$

It is also possible to take (12) and do the cube root:

16) $$ \sqrt[3]{\Delta sal_{out}} = \sqrt[3]{\frac{flow_{local} (\Delta sal_{local})}{flow_{out}}} $$

## Parameterized Curve Fitting

Approximate the c_local using parameterized curve fitting.

17) $$ c_{out} = \frac{c_{in} \cdot flow_{in} + c_{local} \cdot flow_{local}}{flow_{out}} $$
