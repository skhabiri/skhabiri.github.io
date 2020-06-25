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
```

'<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>name</th>\n      <th>MDVP:Fo(Hz)</th>\n      <th>MDVP:Fhi(Hz)</th>\n      <th>MDVP:Flo(Hz)</th>\n      <th>MDVP:Jitter(%)</th>\n      <th>MDVP:Jitter(Abs)</th>\n      <th>MDVP:RAP</th>\n      <th>MDVP:PPQ</th>\n      <th>Jitter:DDP</th>\n      <th>MDVP:Shimmer</th>\n      <th>MDVP:Shimmer(dB)</th>\n      <th>Shimmer:APQ3</th>\n      <th>Shimmer:APQ5</th>\n      <th>MDVP:APQ</th>\n      <th>Shimmer:DDA</th>\n      <th>NHR</th>\n      <th>HNR</th>\n      <th>status</th>\n      <th>RPDE</th>\n      <th>DFA</th>\n      <th>spread1</th>\n      <th>spread2</th>\n      <th>D2</th>\n      <th>PPE</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>phon_R01_S01_1</td>\n      <td>119.992</td>\n      <td>157.302</td>\n      <td>74.997</td>\n      <td>0.00784</td>\n      <td>0.00007</td>\n      <td>0.00370</td>\n      <td>0.00554</td>\n      <td>0.01109</td>\n      <td>0.04374</td>\n      <td>0.426</td>\n      <td>0.02182</td>\n      <td>0.03130</td>\n      <td>0.02971</td>\n      <td>0.06545</td>\n      <td>0.02211</td>\n      <td>21.033</td>\n      <td>1</td>\n      <td>0.414783</td>\n      <td>0.815285</td>\n      <td>-4.813031</td>\n      <td>0.266482</td>\n      <td>2.301442</td>\n      <td>0.284654</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>phon_R01_S01_2</td>\n      <td>122.400</td>\n      <td>148.650</td>\n      <td>113.819</td>\n      <td>0.00968</td>\n      <td>0.00008</td>\n      <td>0.00465</td>\n      <td>0.00696</td>\n      <td>0.01394</td>\n      <td>0.06134</td>\n      <td>0.626</td>\n      <td>0.03134</td>\n      <td>0.04518</td>\n      <td>0.04368</td>\n      <td>0.09403</td>\n      <td>0.01929</td>\n      <td>19.085</td>\n      <td>1</td>\n      <td>0.458359</td>\n      <td>0.819521</td>\n      <td>-4.075192</td>\n      <td>0.335590</td>\n      <td>2.486855</td>\n      <td>0.368674</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>phon_R01_S01_3</td>\n      <td>116.682</td>\n      <td>131.111</td>\n      <td>111.555</td>\n      <td>0.01050</td>\n      <td>0.00009</td>\n      <td>0.00544</td>\n      <td>0.00781</td>\n      <td>0.01633</td>\n      <td>0.05233</td>\n      <td>0.482</td>\n      <td>0.02757</td>\n      <td>0.03858</td>\n      <td>0.03590</td>\n      <td>0.08270</td>\n      <td>0.01309</td>\n      <td>20.651</td>\n      <td>1</td>\n      <td>0.429895</td>\n      <td>0.825288</td>\n      <td>-4.443179</td>\n      <td>0.311173</td>\n      <td>2.342259</td>\n      <td>0.332634</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>phon_R01_S01_4</td>\n      <td>116.676</td>\n      <td>137.871</td>\n      <td>111.366</td>\n      <td>0.00997</td>\n      <td>0.00009</td>\n      <td>0.00502</td>\n      <td>0.00698</td>\n      <td>0.01505</td>\n      <td>0.05492</td>\n      <td>0.517</td>\n      <td>0.02924</td>\n      <td>0.04005</td>\n      <td>0.03772</td>\n      <td>0.08771</td>\n      <td>0.01353</td>\n      <td>20.644</td>\n      <td>1</td>\n      <td>0.434969</td>\n      <td>0.819235</td>\n      <td>-4.117501</td>\n      <td>0.334147</td>\n      <td>2.405554</td>\n      <td>0.368975</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>phon_R01_S01_5</td>\n      <td>116.014</td>\n      <td>141.781</td>\n      <td>110.655</td>\n      <td>0.01284</td>\n      <td>0.00011</td>\n      <td>0.00655</td>\n      <td>0.00908</td>\n      <td>0.01966</td>\n      <td>0.06425</td>\n      <td>0.584</td>\n      <td>0.03490</td>\n      <td>0.04825</td>\n      <td>0.04465</td>\n      <td>0.10470</td>\n      <td>0.01767</td>\n      <td>19.649</td>\n      <td>1</td>\n      <td>0.417356</td>\n      <td>0.823484</td>\n      <td>-3.747787</td>\n      <td>0.234513</td>\n      <td>2.332180</td>\n      <td>0.410335</td>\n    </tr>\n  </tbody>\n</table>'

