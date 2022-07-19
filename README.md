# SSP-MMC-Plus

Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory

## Dataset and Format

The dataset is available on [Dataverse](https://doi.org/10.7910/DVN/VAGUL0) (1.6 GB). This is a 7zipped TSV file containing our experiments' 220 million MaiMemo student memory behavior logs.

The columns are as follows:

u - student user ID who reviewed the word (anonymized)

w - spelling of the word

i - total times the user has reviewed the word

d - difficulty of the word

t_history - interval sequence of the historic reviews

r_history - recall sequence of the historic reviews

delta_t - time elapsed from the last review

r - result of the review

p_recall - probability of recall

total_cnt - number of users who did the same memory behavior
