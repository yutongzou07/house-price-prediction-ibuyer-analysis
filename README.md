# house-price-prediction-ibuyer-analysis
# House Price Prediction: Deep Learning & iBuyer Profit Analysis

**PyTorch regression pipeline** predicting home sale prices, with a profit-viability analysis of an iBuyer (instant home-buying) business model built on top of the predictions.

*By Yutong Zou and Sannidhi Patel.*

[Open in Colab](#) *(update this link once your notebook is pushed)*

## Business Problem

Two linked tasks: (1) build the most accurate possible model to predict home sale prices from property features, competing on a class Kaggle leaderboard against classmates' models, and (2) use those predictions to evaluate whether an iBuyer business model (think Zillow Offers) — which buys homes directly from sellers at an algorithmically-set price, then resells them — would actually be profitable.

## Approach

**1. Data preparation**
Cleaned and explored a residential sales dataset (price distribution, neighborhood-level counts, feature correlations), one-hot encoded categorical features, standardized numeric features, and imputed missing values. Sale price was normalized by a factor of 100,000 to stabilize training.

**2. Iterative model development**
Built and compared four increasingly sophisticated architectures, using **Median Error Rate (MER)** — the median absolute percentage error between predicted and actual price — as the evaluation metric:

| Model | Architecture | Validation MER |
|---|---|---|
| Linear regression | Single linear layer | baseline |
| Base MLP | 2 hidden layers (256, 128), ReLU | ~0.107 (10.7%) |
| Deep MLP | 4 hidden layers (512, 256, 128, 64), ReLU | ~0.107 (10.7%) |
| Deep MLP + L2 regularization | Same 4-layer architecture, weight decay | ~0.104 (10.4%) |
| **Final model** | 4-layer MLP + BatchNorm + LeakyReLU + Dropout + AdamW | **0.081 (8.1%)** |

The final architecture — adding batch normalization, LeakyReLU activations, and dropout on top of L2 regularization — meaningfully outperformed the simpler MLPs, cutting validation error by roughly 25% relative to the base model. This ranked **#6 in the class** on the private Kaggle leaderboard for prediction accuracy.

**3. iBuyer profit analysis**
Applied the trained model's predictions to simulate an iBuyer business model:
- Offer price set at `predicted_price / (1 + target_margin)`, with a 12% target margin
- Homeowner acceptance modeled via a discount-threshold rule (accept if offer > 90% of actual sale price)
- Evaluated realized profit under both a naive "all offers accepted" scenario and a realistic acceptance-rate scenario

## Key Findings

- **If every offer were accepted regardless of price**, the average realized profit margin was **13.6%** — slightly above the 12% target, since predictions were on average slightly conservative (positive bias of ~3%).
- **Under a realistic acceptance rule**, only **~45% of offers** were actually accepted by simulated sellers. Among those accepted offers, average profit margin dropped to **-2.6%** — well below the 12% target, and in fact slightly loss-making.
- This gap reveals a **classic adverse selection problem**: sellers were more likely to accept offers when the model had *overpredicted* the home's value (mean prediction bias among accepted offers was +19.1%, much higher than the ~3% overall bias). In other words, the iBuyer disproportionately wins the properties it overvalued and loses the bids where its estimate was fair or conservative — a critical risk for this business model that isn't visible from headline accuracy metrics alone.

## Recommendation / Business Impact

Prediction accuracy alone (MER) is not sufficient to evaluate an iBuyer strategy — the **interaction between prediction error and seller acceptance behavior** can silently erode margins even when the underlying model is accurate on average. A real-world deployment would need either tighter uncertainty-aware pricing (e.g. discounting offers more heavily when the model is less confident) or a mechanism to detect and avoid systematically overvalued properties before they're purchased.

## Tools

Python, PyTorch (custom MLP architectures, training loops, checkpointing), pandas, scikit-learn (preprocessing, train/validation split), matplotlib/seaborn

## Notes

This was a team project for a graduate machine learning course, run as a class-wide Kaggle competition. Dataset is proprietary to the course and not included in this repository.
