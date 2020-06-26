---
layout: post
title: Step by Step Data Analysis and Visualization
#subtitle: Oxford Parkinson's Disease Detection Dataset
gh-repo: https://github.com/skhabiri/DS-Unit-1-Build
gh-badge: [star, fork, follow]
tags: [Data Analysis]
comments: true
---

This post shows a step by step process to explore and select significant features in Oxford Parkinson's Disease Detection Dataset.

* **Dataset:** https://archive.ics.uci.edu/ml/datasets/Parkinsons

* **Dataset information:**
This dataset is composed of a range of biomedical voice measurements from people with Parkinson's disease (PD) or without PD. Each column in the table is a particular voice measure, and each row corresponds one of voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.
> 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

## *Loading dataset*


```python
import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data_set = pd.read_csv(url)

## *Exploring and Cleaning*
Initial data exploration gives us some insight about the size of dataset, data types and class labels. Looking at our data reveals that all the features are numeric with a binary class label named "status". We remove the "name" column as it doesn't contain useful information for the purpose of our analysis.

data_set.head()
```
(195, 24)
<table border="1" class="dataframe" style="overflow-x: scroll;display: block;"><thead><tr style="text-align: right;"><th></th><th>name</th><th>MDVP:Fo(Hz)</th><th>MDVP:Fhi(Hz)</th><th>MDVP:Flo(Hz)</th><th>MDVP:Jitter(%)</th><th>MDVP:Jitter(Abs)</th><th>MDVP:RAP</th><th>MDVP:PPQ</th><th>Jitter:DDP</th><th>MDVP:Shimmer</th><th>MDVP:Shimmer(dB)</th><th>Shimmer:APQ3</th><th>Shimmer:APQ5</th><th>MDVP:APQ</th><th>Shimmer:DDA</th><th>NHR</th><th>HNR</th><th>status</th><th>RPDE</th><th>DFA</th><th>spread1</th><th>spread2</th><th>D2</th><th>PPE</th></tr></thead><tbody><tr><th>0</th><td>phon_R01_S01_1</td><td>119.992</td><td>157.302</td><td>74.997</td><td>0.00784</td><td>0.00007</td><td>0.00370</td><td>0.00554</td><td>0.01109</td><td>0.04374</td><td>0.426</td><td>0.02182</td><td>0.03130</td><td>0.02971</td><td>0.06545</td><td>0.02211</td><td>21.033</td><td>1</td><td>0.414783</td><td>0.815285</td><td>-4.813031</td><td>0.266482</td><td>2.301442</td><td>0.284654</td></tr><tr><th>1</th><td>phon_R01_S01_2</td><td>122.400</td><td>148.650</td><td>113.819</td><td>0.00968</td><td>0.00008</td><td>0.00465</td><td>0.00696</td><td>0.01394</td><td>0.06134</td><td>0.626</td><td>0.03134</td><td>0.04518</td><td>0.04368</td><td>0.09403</td><td>0.01929</td><td>19.085</td><td>1</td><td>0.458359</td><td>0.819521</td><td>-4.075192</td><td>0.335590</td><td>2.486855</td><td>0.368674</td></tr><tr><th>2</th><td>phon_R01_S01_3</td><td>116.682</td><td>131.111</td><td>111.555</td><td>0.01050</td><td>0.00009</td><td>0.00544</td><td>0.00781</td><td>0.01633</td><td>0.05233</td><td>0.482</td><td>0.02757</td><td>0.03858</td><td>0.03590</td><td>0.08270</td><td>0.01309</td><td>20.651</td><td>1</td><td>0.429895</td><td>0.825288</td><td>-4.443179</td><td>0.311173</td><td>2.342259</td><td>0.332634</td></tr><tr><th>3</th><td>phon_R01_S01_4</td><td>116.676</td><td>137.871</td><td>111.366</td><td>0.00997</td><td>0.00009</td><td>0.00502</td><td>0.00698</td><td>0.01505</td><td>0.05492</td><td>0.517</td><td>0.02924</td><td>0.04005</td><td>0.03772</td><td>0.08771</td><td>0.01353</td><td>20.644</td><td>1</td><td>0.434969</td><td>0.819235</td><td>-4.117501</td><td>0.334147</td><td>2.405554</td><td>0.368975</td></tr><tr><th>4</th><td>phon_R01_S01_5</td><td>116.014</td><td>141.781</td><td>110.655</td><td>0.01284</td><td>0.00011</td><td>0.00655</td><td>0.00908</td><td>0.01966</td><td>0.06425</td><td>0.584</td><td>0.03490</td><td>0.04825</td><td>0.04465</td><td>0.10470</td><td>0.01767</td><td>19.649</td><td>1</td><td>0.417356</td><td>0.823484</td><td>-3.747787</td><td>0.234513</td><td>2.332180</td><td>0.410335</td></tr></tbody></table>


```python
data_set.dtypes
```
![img1](/assets/img/post1_dtype.png)
