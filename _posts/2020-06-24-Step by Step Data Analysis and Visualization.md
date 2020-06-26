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
data_set = pd.read_csv(url)```

## *Exploring and Cleaning*
Initial data exploration gives us some insight about the size of dataset, data types and class labels. Looking at our data reveals that all the features are numeric with a binary class label named "status". We remove the "name" column as it doesn't contain useful information for the purpose of our analysis.

```python
print(data_set.shape)
data_set.head()
```

(195, 24)
<table border="1" class="dataframe" style="overflow-x: scroll;display: block;"><thead><tr style="text-align: right;"><th></th><th>name</th><th>MDVP:Fo(Hz)</th><th>MDVP:Fhi(Hz)</th><th>MDVP:Flo(Hz)</th><th>MDVP:Jitter(%)</th><th>MDVP:Jitter(Abs)</th><th>MDVP:RAP</th><th>MDVP:PPQ</th><th>Jitter:DDP</th><th>MDVP:Shimmer</th><th>MDVP:Shimmer(dB)</th><th>Shimmer:APQ3</th><th>Shimmer:APQ5</th><th>MDVP:APQ</th><th>Shimmer:DDA</th><th>NHR</th><th>HNR</th><th>status</th><th>RPDE</th><th>DFA</th><th>spread1</th><th>spread2</th><th>D2</th><th>PPE</th></tr></thead><tbody><tr><th>0</th><td>phon_R01_S01_1</td><td>119.992</td><td>157.302</td><td>74.997</td><td>0.00784</td><td>0.00007</td><td>0.00370</td><td>0.00554</td><td>0.01109</td><td>0.04374</td><td>0.426</td><td>0.02182</td><td>0.03130</td><td>0.02971</td><td>0.06545</td><td>0.02211</td><td>21.033</td><td>1</td><td>0.414783</td><td>0.815285</td><td>-4.813031</td><td>0.266482</td><td>2.301442</td><td>0.284654</td></tr><tr><th>1</th><td>phon_R01_S01_2</td><td>122.400</td><td>148.650</td><td>113.819</td><td>0.00968</td><td>0.00008</td><td>0.00465</td><td>0.00696</td><td>0.01394</td><td>0.06134</td><td>0.626</td><td>0.03134</td><td>0.04518</td><td>0.04368</td><td>0.09403</td><td>0.01929</td><td>19.085</td><td>1</td><td>0.458359</td><td>0.819521</td><td>-4.075192</td><td>0.335590</td><td>2.486855</td><td>0.368674</td></tr><tr><th>2</th><td>phon_R01_S01_3</td><td>116.682</td><td>131.111</td><td>111.555</td><td>0.01050</td><td>0.00009</td><td>0.00544</td><td>0.00781</td><td>0.01633</td><td>0.05233</td><td>0.482</td><td>0.02757</td><td>0.03858</td><td>0.03590</td><td>0.08270</td><td>0.01309</td><td>20.651</td><td>1</td><td>0.429895</td><td>0.825288</td><td>-4.443179</td><td>0.311173</td><td>2.342259</td><td>0.332634</td></tr><tr><th>3</th><td>phon_R01_S01_4</td><td>116.676</td><td>137.871</td><td>111.366</td><td>0.00997</td><td>0.00009</td><td>0.00502</td><td>0.00698</td><td>0.01505</td><td>0.05492</td><td>0.517</td><td>0.02924</td><td>0.04005</td><td>0.03772</td><td>0.08771</td><td>0.01353</td><td>20.644</td><td>1</td><td>0.434969</td><td>0.819235</td><td>-4.117501</td><td>0.334147</td><td>2.405554</td><td>0.368975</td></tr><tr><th>4</th><td>phon_R01_S01_5</td><td>116.014</td><td>141.781</td><td>110.655</td><td>0.01284</td><td>0.00011</td><td>0.00655</td><td>0.00908</td><td>0.01966</td><td>0.06425</td><td>0.584</td><td>0.03490</td><td>0.04825</td><td>0.04465</td><td>0.10470</td><td>0.01767</td><td>19.649</td><td>1</td><td>0.417356</td><td>0.823484</td><td>-3.747787</td><td>0.234513</td><td>2.332180</td><td>0.410335</td></tr></tbody></table>


```python
data_set.dtypes
```
![img1](/assets/img/post1_dtype.png)

We split our dataset into data X and label y.

```python
X = data_set.iloc[:,1:].drop(labels="status", axis=1)
y = data_set["status"]
print(f'data: {X.shape}\nlabel: {y.shape}')
```

data: (195, 22)
label: (195,)

```python
y.value_counts()
print(f'Number of patients diagnosed with PD: {y.value_counts()[1]}')
print(f'Number of patients without PD: {y.value_counts()[0]}')
```

Number of patients diagnosed with PD: 147
Number of patients without PD: 48

```python
X.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="overflow-x: scroll;display: block;">
      <th></th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:PPQ</th>
      <th>MDVP:APQ</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>154.228641</td>
      <td>197.104918</td>
      <td>116.324631</td>
      <td>0.000044</td>
      <td>0.003446</td>
      <td>0.024081</td>
      <td>0.024847</td>
      <td>21.885974</td>
      <td>0.498536</td>
      <td>0.718099</td>
      <td>0.226510</td>
      <td>2.381826</td>
      <td>0.206552</td>
    </tr>
    <tr>
      <th>std</th>
      <td>41.390065</td>
      <td>91.491548</td>
      <td>43.521413</td>
      <td>0.000035</td>
      <td>0.002759</td>
      <td>0.016947</td>
      <td>0.040418</td>
      <td>4.425764</td>
      <td>0.103942</td>
      <td>0.055336</td>
      <td>0.083406</td>
      <td>0.382799</td>
      <td>0.090119</td>
    </tr>
    <tr>
      <th>min</th>
      <td>88.333000</td>
      <td>102.145000</td>
      <td>65.476000</td>
      <td>0.000007</td>
      <td>0.000920</td>
      <td>0.007190</td>
      <td>0.000650</td>
      <td>8.441000</td>
      <td>0.256570</td>
      <td>0.574282</td>
      <td>0.006274</td>
      <td>1.423287</td>
      <td>0.044539</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.572000</td>
      <td>134.862500</td>
      <td>84.291000</td>
      <td>0.000020</td>
      <td>0.001860</td>
      <td>0.013080</td>
      <td>0.005925</td>
      <td>19.198000</td>
      <td>0.421306</td>
      <td>0.674758</td>
      <td>0.174351</td>
      <td>2.099125</td>
      <td>0.137451</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>148.790000</td>
      <td>175.829000</td>
      <td>104.315000</td>
      <td>0.000030</td>
      <td>0.002690</td>
      <td>0.018260</td>
      <td>0.011660</td>
      <td>22.085000</td>
      <td>0.495954</td>
      <td>0.722254</td>
      <td>0.218885</td>
      <td>2.361532</td>
      <td>0.194052</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>182.769000</td>
      <td>224.205500</td>
      <td>140.018500</td>
      <td>0.000060</td>
      <td>0.003955</td>
      <td>0.029400</td>
      <td>0.025640</td>
      <td>25.075500</td>
      <td>0.587562</td>
      <td>0.761881</td>
      <td>0.279234</td>
      <td>2.636456</td>
      <td>0.252980</td>
    </tr>
    <tr>
      <th>max</th>
      <td>260.105000</td>
      <td>592.030000</td>
      <td>239.170000</td>
      <td>0.000260</td>
      <td>0.019580</td>
      <td>0.137780</td>
      <td>0.314820</td>
      <td>33.047000</td>
      <td>0.685151</td>
      <td>0.825288</td>
      <td>0.450493</td>
      <td>3.671155</td>
      <td>0.527367</td>
    </tr>
  </tbody>
</table>

