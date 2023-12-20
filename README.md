# SSP-MMC-Plus

Copyright (c) 2023 [MaiMemo](https://www.maimemo.com/), Inc. MIT License.

SSP-MMC-Plus is the extended version of [SSP-MMC](https://github.com/maimemo/SSP-MMC), a spaced repetition scheduling algorithm used to help learners remember more words in MaiMemo, a language learning application in China.

This repository contains a public release of the data and code used for several experiments in the following paper (which introduces SSP-MMC-Plus):

> J. Su, J. Ye, L. Nie, Y. Cao and Y. Chen, "Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2023.3251721.

The paper is a substantial extension of our previous conference paper [A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling](https://www.maimemo.com/paper/) (free access).

When using this dataset and/or code, please cite this publication. A BibTeX record is:

```
@ARTICLE{10059206,
  author={Su, Jingyong and Ye, Junyao and Nie, Liqiang and Cao, Yilong and Chen, Yongyong},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Optimizing Spaced Repetition Schedule by Capturing the Dynamics of Memory}, 
  year={2023},
  volume={35},
  number={10},
  pages={10085-10097},
  doi={10.1109/TKDE.2023.3251721}}
```

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
