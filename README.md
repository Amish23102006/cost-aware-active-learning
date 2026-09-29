# When Does Cost-Aware Active Learning Help?

An empirical study of cost heterogeneity and budget-constrained sample selection in active learning for text classification.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Amish23102006/cost-aware-active-learning/blob/main/cost_aware_active_learning.ipynb)

> **Status:** research draft. The accompanying paper is under review and not yet submission-ready.

## Overview

Most active learning (AL) research assumes every sample costs the same to label. In practice, annotation cost varies (for example, longer texts take longer to read). This project asks how much of the benefit of cost-aware AL can be captured by simple heuristics, without Bayesian batch acquisition or heavy optimization machinery.

Two selection strategies are compared against plain uncertainty sampling and random sampling:

- **Greedy cost-aware ranking:** score(x) = u(x) / c(x)^λ, where u(x) is model uncertainty, c(x) is annotation cost, and λ controls the cost penalty (λ = 0 recovers plain uncertainty sampling).
- **0-1 knapsack selection:** choose the batch that maximizes total uncertainty subject to a cost budget, using only standard (non-Bayesian) uncertainty scores. Candidates are restricted each round to the top 5x-batch-size samples by uncertainty.

## Setup

- **Datasets:** AG News (4-class topic classification) and IMDB (binary sentiment), subsampled to 6,000 train / 2,000 test examples.
- **Model:** TF-IDF features (5,000 max features) with logistic regression, retrained each round. A robustness check uses frozen `all-MiniLM-L6-v2` sentence embeddings.
- **Cost model:** simulated, length-based: `cost(x) = base_cost + length_factor * word_count(x)`.
- **Protocol:** 15 AL rounds, 10 random seeds, with paired bootstrap 95% confidence intervals on cost-to-target (the cost needed to reach 98% of peak accuracy).

## Key findings

1. **Greedy cost weighting:** moderate λ values show no statistically robust improvement over plain uncertainty sampling on AG News, but fully weighting by cost (λ = 1.0) is reliably harmful.
2. **Knapsack selection has no uniform benefit:** it is significantly worse than plain uncertainty on low-variance AG News and significantly better on high-variance IMDB.
3. **Cost variance is the driver:** in a controlled experiment that manipulates cost variance on fixed data, the cost-aware advantage grows with cost heterogeneity and becomes statistically significant at moderate-to-high variance (CV >= 0.5).
4. **Robust to representation:** the core AL benefit persists when TF-IDF is replaced with pretrained transformer embeddings.

See the paper draft for full tables, confidence intervals, and limitations.

## Running the code

**In Colab (easiest):** click the badge above and run all cells.

**Locally:**

```bash
git clone https://github.com/Amish23102006/cost-aware-active-learning.git
cd cost-aware-active-learning
pip install -r requirements.txt
jupyter notebook cost_aware_active_learning.ipynb
```

## Limitations

- Annotation costs are simulated, not measured from real annotation logs. A small single-annotator timing pilot showed only a weak positive correlation between length and time.
- Only two English text classification datasets and a classical TF-IDF + logistic regression model are used as the main setup.
- The knapsack is exact only within a restricted candidate pool, which may contribute to its AG News disadvantage.

## Related work

This project is complementary to KnapsackBALD (Dossou et al., 2026), which combines BatchBALD acquisition with knapsack optimization. It does not claim to outperform that method; it isolates how much benefit simpler cost-aware heuristics provide.

## Author

Amish Chaturvedi

## License

MIT (add a `LICENSE` file to the repository).
