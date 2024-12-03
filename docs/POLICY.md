# How to do policy

This section is meant to describe the meta-process for how to do analysis with the end goal of creating
policy recommendations. Thoughts are my own, but inspired by the direction of the program advisors.
There are some probably some gaps and things may be inaccurate, but hopefully this gives a look behind the
curtain at how these things go.

Although this is posed as a sequential list, the steps can repeated, backtracked on, or repeated in isolation.

- Prerequisites
  - **Need:** Before there is even a process for analysis, there must be a need to do the analysis. Simply put, this
  need arises from the people affected. In this case it comes from pressures from agricultural producers in
  the LRG/RB. Industries like the sugar cane industry are nonviable now because of the salt levels and yields
  are decreasing nonlinearly for other crops. This pressure comes from both sides and people have died in protest. There
  is obivously a need.
  - **Care:** People who can make decisions must care about the needs of the people. Having a need is not enough, it
  requires that key decision makers in power care about the need. I won't give examples, you are probably aware of
  issues that fail to produce legislation, however in this case in particular. The agencies that hold control over the
  LRG/RB have been incredibly receptive to the needs of their constituents.
  - **Organization:** The key actors must be willing to work together and be organized in a framework that allows for
  policy to be passed. There are six agencies that hold stake in the river. They know who they are and have a long history
  of working together to set meaningful policy.
  - **Experts:** Generally speaking, policy makers contract out work to consultant experts because they don't have the
  bandwidth to do the research themselves. There are a number of people in the space who have done work, but in this case
  there simply wasn't enough capital. This lack of capital required creativity. Our class is a Policy Research Project
  with undergraduate and graduate students from the LBJ School of Public Affairs as well as collaboration with graduate
  students from UDLAP, Mexico. The leadership in the class consists of Dr. Eaton, Dr. Pavon, Dr. Sheer, each of which have
  deep knowledge of water management policy. With this structure, we act as the expert consultants for the agencies, working
  intimately with leadership. All results are vetted, of course, so the quality is not degraded, but the results are
  produced for much less money since there is an implicit exchange for the opportunity to learn rather than pay.
- Data and modeling
  - **Get the data sources:** Getting the data is the hardest part. In this study we are working with four tiers of data.
  Official data agreed upon by the agencies, non-official data that is gotten from external sources, inferred data such
  interpolation over gaps, and unknowns. Each tier leads to a decrease in confidence in the results with ideal case being
  that all official data is complete an accurate for the time range for the study. In our case, this was mostly complete,
  however certain data has been swapped (HUC Rainfall instead of precipitation on the reach), interpolation for gaps in the
  data (taking average values or linear scales over short timeframes), and some data was missing completely requiring a
  remodeling (there is no salinity data for below the Anzalduas Dam).
  - **Clean the data:** Mentioned above, but there is a strong requirement for data cleaning. This involves the interpolation
  of missing data, the formatting of null values based on software tolerance (OASIS famously takes decimals values only), the
  verification of data collection (are gauges spitting out nonsense values?). Just to name a few things. This is a labor
  intensive process and requires constant updates as new information comes to light. This is probably three quarters of
  all the hours put into the project.
  - **Build the model:** Dan has said, "all models are wrong, some are useful." When building a model, it is important to
  take into the context the available data, the organizational structure, and the ultimate goal based on the needs. For
  this project in particular we went with a simple flow and salt mass balance model. Although this doesn't capture all
  behavior in the most incredible amount of detail, it provides enough insight into the problem with the data we have
  so that we can build effective policy recommendations.
- Analysis
  - **Format output for custom analysis:** This is a halfway step. Depending on whatever piece of antiquated software you
  are using for the model (there really aren't a ton of modern, open-source hydrology softwares for mass balance), it may
  be necessary to export the model output so that better analysis can be done.
  - **Graphical analysis:** This was a bit of a paradigm shift for me. I had previously thought that analysis began with
  a generalized statistical analysis upon all the features in the dataset to reveal clairvoyant insights. Thanks Big Data.
  For a study like this, what is tantamount is the ability to visually identify behaviors and corroborate the behaviors with
  external references. For example, noticing persistent water loss within a reach to some external factor and then backing
  it up with a study that shows the LRG/RB being a losing stream to groundwater over that reach. Having domain knowledge,
  and doing graphical analysis provides a basis for establishing causality, not just discovering correlation. From this
  hypotheses can be formed.
  - **Event analysis:** Depending on the reliability of the dataset, sometimes generalized trends can't be inferred and
  rather single events matter. You still build hypotheses, but the way the regression analysis is performed looks different.
  An example would be salinity spikes after rainfall events and backing this up with a weather report.
  - **Statistical analysis:** This is a supplemental analysis and involves making basic statistical plots and metrics to
  describe and back up behavior identified in the graphical analysis. As a side note, box plots are amazing.
- Scientific Method
  - **Establish hypotheses:** This section follows closely to the scientific method that we all learned in grade school.
  Ask a question, build a hypothesis, establish correlation that supports or rejects the null hypothesis, derive conclusions.
  I'll be pretty brief here and only note the key points as they pertain to this study in particular.
  - **Regression analysis:** Establish a correlation. For hydrology, you are rarely working within the context of linear
  relationships. Again, this is where domain knowledge comes into play. Taking the Pearson correlation of two variables
  will not lead to anything conclusive. Understand the nuances that come with time series, anomalies, sparse datasets, and
  nonlinear relationships. You can still establish a margin of error for a predicted value with an observed value outside
  of the basic statistical functions that are available, in fact it is necessary.
  - **Interpret results:** Only one thing here to note, sometimes no result is a result. "Confirm", "deny", or "can't say" are
  all valuable for building policy.
- Policy
  - **Recommendations:** Admittedly, this is the part I know the least about. What it appear to be is; take the results,
  understand the solution space, find recommendations that fit the results and back up the recommendations with references
  from history or other papers. People also really seem to care about costs, so be sure of those. Understand that
  there is no singular recommendation that works, instead it is a cost-benefit tradeoff between all available recommendations
  that the key actors can then select from, amend, or reject. The most important part is making sure that the recommendations
  are well-supported.

General notes:
- Building policy is just like a job. It requires a few key people to drive the project, clear communication and having respect
and shared values among everyone. Disorganization is easy when working in academia, rebel against the urge and be organized.
- Sometimes "need more data" is a perfectly valid policy recommendation.
- Understand the political context beyond the scope of work, at the end of the day, it dictates whether or not certain people
will be willing to work together at all.
- Understand the incentives that people have so that recommendations and research methods can be tailored effectively.
