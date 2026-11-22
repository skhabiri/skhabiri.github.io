---
layout: post
title: "Why 99.9% Accuracy Can Mean a Useless Fraud Model"
subtitle: Classifying severely imbalanced data for fraud detection
tags: [Machine Learning, Fraud Detection, Imbalanced Data, Classification]
comments: false
---

By Shawn Khabiri

Fraud is rare. In most payment data, well under 1% of transactions are fraudulent. That single fact changes how you measure a model, how you train it and where you set its threshold.

Every fraud model makes two kinds of mistakes: false alarms and missed cases. This article walks through the decisions that trade one against the other, using one example. The techniques are standard and public. The hard part is knowing which one to use and what it costs you.

## Start with the numbers

Take one million transactions. 1,000 are fraud and 999,000 are legitimate. The fraud rate is 0.1%. These numbers are made up for illustration.

Now consider a model that marks every transaction as legitimate. It is right 999,000 times out of a million. Its accuracy is 99.9%. It catches no fraud at all.

Accuracy fails here because the legitimate class dominates the count. Any metric that rewards getting the majority right will look good no matter what happens to the fraud cases.

## Precision and recall

Two questions matter more than accuracy.

- Recall: of all the fraud, how much did the model catch? Low recall means missed cases.
- Precision: of all the alerts, how many were real fraud? Low precision means false alarms.

A fraud model outputs a score, and a threshold turns the score into an alert. Here is the same model at two thresholds.

| | Threshold A (strict) | Threshold B (loose) |
| --- | --- | --- |
| Fraud caught | 600 | 850 |
| Fraud missed | 400 | 150 |
| False alarms | 400 | 7,650 |
| Total alerts | 1,000 | 8,500 |
| Recall | 60% | 85% |
| Precision | 60% | 10% |
| Accuracy | 99.92% | 99.22% |

Look at the last row. Threshold B catches 250 more fraud cases, and its accuracy is lower. Both sit within a point of the model that catches nothing.

F1 combines precision and recall into one number: 0.60 for A and 0.18 for B. But F1 treats a missed case and a false alarm as equally bad. In fraud they almost never are.

## Why the precision-recall curve beats ROC here

The ROC curve plots recall against the false positive rate. At threshold B the false positive rate is 7,650 out of 999,000, or 0.77%. That looks excellent. Yet nine out of ten alerts are wrong.

The reason is true negatives. The false positive rate is measured against all legitimate transactions, and almost all of them are easy calls. Precision and recall leave true negatives out entirely. For rare events, that makes the precision-recall curve the more honest picture.

Baselines differ too. A random model scores 0.5 on ROC AUC. On the precision-recall curve its baseline is the fraud rate, here 0.001. An average precision of 0.30 sounds weak, and it is 300 times better than chance.

## The threshold is a separate decision

Classifiers default to a cutoff of 0.5. When fraud is rare, the scores for real fraud cases may never get that high. Lowering the cutoff to 0.3 or 0.2 catches more fraud and raises more false alarms.

Two things make the threshold worth treating separately. It needs no retraining, so you can test any cutoff on validation data in seconds. And moving it does not change the area under the ROC or precision-recall curve. Those only move when the model changes. A better curve means a better model. A better threshold means better use of the same model.

The right cutoff is the one with the lowest total cost: the cost of a false alarm times the number of false alarms, plus the cost of a missed case times the number of missed cases. Assume a missed fraud costs $500 on average and reviewing an alert costs $5.

| Option | Missed fraud | Review cost | Total |
| --- | --- | --- | --- |
| Catch nothing | $500,000 | $0 | $500,000 |
| Threshold A | $200,000 | $5,000 | $205,000 |
| Threshold B | $75,000 | $42,500 | $117,500 |

Threshold B wins by a wide margin, with 10% precision. A team that picked the threshold by F1 would have chosen A and paid $87,500 more per million transactions.

Capacity matters as much as cost. If the review team can handle 2,000 alerts per million transactions, threshold B is not available. The question becomes how much fraud sits in the top 2,000 scores.

## Anomaly detection or imbalanced classification?

These two get mixed up. The question that separates them: does fraud form a class you can learn?

Imbalanced classification is supervised. You have labeled fraud cases. They are few, but they form a distinct class, and a model can learn its boundary. Anomaly detection usually works without labels. It learns what normal looks like and flags whatever deviates. Isolation forests, one-class SVMs and autoencoders are common choices.

With reliable labels, the supervised model usually wins, because unusual is not the same as fraudulent. A large purchase on vacation is unusual. A small test charge on a stolen card looks ordinary. Anomaly detection earns its place when labels are missing, or when new fraud patterns appear that the labels do not cover yet.

## Resampling: the direction depends on data size

Standard training favors the majority class and can ignore the minority. Resampling changes the training mix. Which way to go depends mostly on how much data you have.

Oversample when the dataset is small. With 1,000 rows and 50 fraud cases, you cannot afford to drop anything. Repeat the fraud rows, or use SMOTE to create synthetic ones by interpolating between real fraud cases. The risk is overfitting: the model memorizes noise in those 50 cases.

Undersample when the dataset is huge. With 10 million rows and 9.9 million of them legitimate, much of the majority class is redundant. Dropping a large share cuts training time and often costs little.

Model type matters too. Tree models tolerate duplicated rows well. A duplicate mostly reinforces a split the tree would have made anyway. Linear models are more sensitive, because repeated rows pull the coefficients. For those, synthetic samples are often the better choice.

Balanced bagging is a useful middle path. Each model in the ensemble trains on its own balanced sample, so no single sample has to discard most of the data.

Both directions push the decision boundary into majority territory. You trade precision for recall. Balance a 1-to-100 dataset to 1-to-1 and the model trains in a world where fraud is half of everything. It will then flag fraud far more often than fraud occurs.

So resample the training data only. Keep validation and test data at the real fraud rate, using stratified splits.

## Cost-sensitive learning

The other route leaves the data alone and changes what errors cost. Class weights are the simplest form: each fraud error counts 10 or 100 times more in the loss. No rows are dropped and none are duplicated.

The risk is extreme weights. Measure total cost on validation data. If it is much worse than on training data, the model has fit the weights and not the pattern.

The logic fits in one small calculation. Say a missed fraud costs 10 and a false alarm costs 1. The model gives a transaction a 20% chance of fraud.

- Expected cost of calling it legitimate: 0.2 × 10 = 2.0
- Expected cost of calling it fraud: 0.8 × 1 = 0.8

Flagging is cheaper, even though the transaction is probably legitimate. MetaCost, a wrapper method, uses this calculation to relabel training data so that any standard model learns the cost-aware decision.

The same arithmetic gives a break-even point. With costs of 10 and 1, flag anything above about 9%. With the $500 and $5 from earlier, the break-even is about 1%.

## Fix the probabilities afterward

That calculation needs real probabilities, and both resampling and class weights distort them. A model might score a group of transactions at 0.9 when only half of them are fraud.

Check before you trust the scores. Group validation predictions into bins by score. In the bin around 0.25, about 25% should be fraud. If the bins are off, calibrate: fit a small second model on validation data that maps raw scores back to real probabilities. Platt scaling fits a sigmoid and suits small validation sets. Isotonic regression fits a step function and needs more data, roughly a thousand validation rows or more.

If you only rank alerts for a review queue, you can skip calibration. If you apply costs to probabilities, you cannot.

## The short version

1. Ignore accuracy. Report precision, recall and the precision-recall curve.
2. Treat the threshold as its own decision. Set it from costs and review capacity, not from 0.5 or F1.
3. Use supervised models when you have labels. Use anomaly detection where the labels run out.
4. Oversample small datasets, undersample huge ones, and resample training data only.
5. If you weight or resample, calibrate before using scores as probabilities.

*Shawn Khabiri is an AI engineering lead. His earlier work includes fraud detection at a large U.S. bank. This article covers general, public techniques only.*
