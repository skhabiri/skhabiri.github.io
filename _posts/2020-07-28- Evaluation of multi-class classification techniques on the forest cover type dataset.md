---
layout: post
title: Evaluation of Multi-class Classification Techniques on The Forest Cover Type Dataset
#subtitle: Oxford Parkinson's Disease Detection Dataset
gh-repo: https://github.com/skhabiri/DS-Unit-1-Build
gh-badge: [star, fork, follow]
tags: [Predictive Modeling]
image: /assets/img/post1_cover3.png
comments: true
---

In this post, five different machine learning classifiers are implemented to classify a multi-class target label. The classification techniques are evaluated on the forest cover type dataset provided by Jock A. Blackard and Colorado State University. 

* **Dataset:** [https://archive.ics.uci.edu/ml/datasets/Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype)

* **Dataset information:**
The dataset was obtained from the University of California, Irvine, School of Information and Computer Sciences database. It contains 15120 observations of cover type with 56 columns including the taget label. The data includes 7 categories of different cover types, 10 continuous variables, 4 wilderness areas and 40 binary soil types. It produces in total 54 different variables available for the models.

> Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science

## *Loading dataset*

```python
# Import data
import pandas as pd
import numpy as np

data = pd.read_csv('https://github.com/skhabiri/Forest_cover_type_data/raw/master/train.csv')
print(data.shape)
data.head()
```

(15120, 56)
<table border="1" class="dataframe" style="overflow-x: scroll;display: block;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area1</th>
      <th>Wilderness_Area2</th>
      <th>Wilderness_Area3</th>
      <th>Wilderness_Area4</th>
      <th>Soil_Type1</th>
      <th>Soil_Type2</th>
      <th>Soil_Type3</th>
      <th>Soil_Type4</th>
      <th>Soil_Type5</th>
      <th>Soil_Type6</th>
      <th>Soil_Type7</th>
      <th>Soil_Type8</th>
      <th>Soil_Type9</th>
      <th>Soil_Type10</th>
      <th>Soil_Type11</th>
      <th>Soil_Type12</th>
      <th>Soil_Type13</th>
      <th>Soil_Type14</th>
      <th>Soil_Type15</th>
      <th>Soil_Type16</th>
      <th>Soil_Type17</th>
      <th>Soil_Type18</th>
      <th>Soil_Type19</th>
      <th>Soil_Type20</th>
      <th>Soil_Type21</th>
      <th>Soil_Type22</th>
      <th>Soil_Type23</th>
      <th>Soil_Type24</th>
      <th>Soil_Type25</th>
      <th>Soil_Type26</th>
      <th>Soil_Type27</th>
      <th>Soil_Type28</th>
      <th>Soil_Type29</th>
      <th>Soil_Type30</th>
      <th>Soil_Type31</th>
      <th>Soil_Type32</th>
      <th>Soil_Type33</th>
      <th>Soil_Type34</th>
      <th>Soil_Type35</th>
      <th>Soil_Type36</th>
      <th>Soil_Type37</th>
      <th>Soil_Type38</th>
      <th>Soil_Type39</th>
      <th>Soil_Type40</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>

All the columns have int dtype. * The dataset has 15120 rows and 56 columns. This is a multiclass classification with "Cover_Type" as the target label. Wilderness_Area and Soil_Type columns have been encoded with One Hot Encoder. For better visualization, we will convert all the Soil_Types and Wilderness_Areas columns into two categorical features "Soil_Type" and "Wilderness_Area".

```python
data1 = data.copy()
# Encode Soil_Types and Wilderness_Area features into two new features
ohe_bool = []
enc_cols = ["Wilderness_Area", "Soil_Type"]
for idx, val in enumerate(enc_cols):
  val_df = data.filter(regex=val, axis=1)
  print(f"{idx} {val} is ohe? {((val_df.sum(axis=1)) == 1).all()}")
  data1[val] = val_df.dot(val_df.columns)
  # Convert the constructed columns into int
  data1[val] = data1[val].astype('str').str.findall(r"(\d+)").str[-1].astype('int')
  data1 = data1.drop(val_df.columns, axis=1)

# Reorder the columns
data1 = data1.iloc[:,data1.columns!="Cover_Type"].merge(data1["Cover_Type"], left_index=True, right_index=True)
data1.head(), data1.shape
```
(15120, 14)

<table border="1" class="dataframe" style="overflow-x: scroll;display: block;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area</th>
      <th>Soil_Type</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2596</td>
      <td>51</td>
      <td>3</td>
      <td>258</td>
      <td>0</td>
      <td>510</td>
      <td>221</td>
      <td>232</td>
      <td>148</td>
      <td>6279</td>
      <td>1</td>
      <td>29</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2590</td>
      <td>56</td>
      <td>2</td>
      <td>212</td>
      <td>-6</td>
      <td>390</td>
      <td>220</td>
      <td>235</td>
      <td>151</td>
      <td>6225</td>
      <td>1</td>
      <td>29</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2804</td>
      <td>139</td>
      <td>9</td>
      <td>268</td>
      <td>65</td>
      <td>3180</td>
      <td>234</td>
      <td>238</td>
      <td>135</td>
      <td>6121</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2785</td>
      <td>155</td>
      <td>18</td>
      <td>242</td>
      <td>118</td>
      <td>3090</td>
      <td>238</td>
      <td>238</td>
      <td>122</td>
      <td>6211</td>
      <td>1</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2595</td>
      <td>45</td>
      <td>2</td>
      <td>153</td>
      <td>-1</td>
      <td>391</td>
      <td>220</td>
      <td>234</td>
      <td>150</td>
      <td>6172</td>
      <td>1</td>
      <td>29</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


```python
data1.describe()
```
<table border="1" class="dataframe" style="overflow-x: scroll;display: block;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Elevation</th>
      <th>Aspect</th>
      <th>Slope</th>
      <th>Horizontal_Distance_To_Hydrology</th>
      <th>Vertical_Distance_To_Hydrology</th>
      <th>Horizontal_Distance_To_Roadways</th>
      <th>Hillshade_9am</th>
      <th>Hillshade_Noon</th>
      <th>Hillshade_3pm</th>
      <th>Horizontal_Distance_To_Fire_Points</th>
      <th>Wilderness_Area</th>
      <th>Soil_Type</th>
      <th>Cover_Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15120.00000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
      <td>15120.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7560.50000</td>
      <td>2749.322553</td>
      <td>156.676653</td>
      <td>16.501587</td>
      <td>227.195701</td>
      <td>51.076521</td>
      <td>1714.023214</td>
      <td>212.704299</td>
      <td>218.965608</td>
      <td>135.091997</td>
      <td>1511.147288</td>
      <td>2.800397</td>
      <td>19.171362</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4364.91237</td>
      <td>417.678187</td>
      <td>110.085801</td>
      <td>8.453927</td>
      <td>210.075296</td>
      <td>61.239406</td>
      <td>1325.066358</td>
      <td>30.561287</td>
      <td>22.801966</td>
      <td>45.895189</td>
      <td>1099.936493</td>
      <td>1.119832</td>
      <td>12.626960</td>
      <td>2.000066</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1863.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-146.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3780.75000</td>
      <td>2376.000000</td>
      <td>65.000000</td>
      <td>10.000000</td>
      <td>67.000000</td>
      <td>5.000000</td>
      <td>764.000000</td>
      <td>196.000000</td>
      <td>207.000000</td>
      <td>106.000000</td>
      <td>730.000000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7560.50000</td>
      <td>2752.000000</td>
      <td>126.000000</td>
      <td>15.000000</td>
      <td>180.000000</td>
      <td>32.000000</td>
      <td>1316.000000</td>
      <td>220.000000</td>
      <td>223.000000</td>
      <td>138.000000</td>
      <td>1256.000000</td>
      <td>3.000000</td>
      <td>17.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11340.25000</td>
      <td>3104.000000</td>
      <td>261.000000</td>
      <td>22.000000</td>
      <td>330.000000</td>
      <td>79.000000</td>
      <td>2270.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>167.000000</td>
      <td>1988.250000</td>
      <td>4.000000</td>
      <td>30.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15120.00000</td>
      <td>3849.000000</td>
      <td>360.000000</td>
      <td>52.000000</td>
      <td>1343.000000</td>
      <td>554.000000</td>
      <td>6890.000000</td>
      <td>254.000000</td>
      <td>254.000000</td>
      <td>248.000000</td>
      <td>6993.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>

Next we plot the  Heatmap of correlation matrix. It shows Elevation and Soil_Type are highly correlated, and Hillshade_3pm is reversely correlated with Hillshade_9am. 

<img src= "/assets/img/post2_heatmap.png">

Looking at scatter plot of different features, it seems "Elevation" is an important parameter in separating different classes of target label.


<p float="left">
  <img src="/assets/img/post2_pairplot1.png" width="450" />
      <img src="/assets/img/post2_pairplot2.png" width="450" /> 
</p>

We make the following observations from the Count plots below. 
* Soil_Type10 shows a significant class distinction for Cover_Type=6.
* Similarly Wilderness_Area4 and Cover_Type=4 are strongly associated.

<img src= "/assets/img/post2_countplot1.png">

List of unique values in the feature shows Soil_Type15 and Soil_Type7 are constant. Additionally, Id column is a unique identifier for each observation. We'll drop those three columns as they do not carry any information to identify the target label.
There are also skewness in some features. "Horizontal_Distance_To_Hydrology" is an example of that.
<img src="/assets/img/post2_skew.png">

### Train-Test Split
After removing "Id" and constant columns, data is split into train and validation set.

```python
from sklearn.model_selection import train_test_split
# Split train into train & val
train, val = train_test_split(data, train_size=0.80, test_size=0.20, stratify=data["Cover_Type"], random_state=42)
# Separate class label and data 
y_train = train["Cover_Type"]
X_train = train.drop("Cover_Type", axis=1)
y_val = val["Cover_Type"]
X_val = val.drop("Cover_Type", axis=1)
print(f'train: {train.shape}, val: {val.shape}')
```
train: (12096, 56), val: (3024, 56)


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
<img src= "/assets/img/post1_describe.png">

## *Handling Missing Data*
A quick check shows our dataset does not have any missing value to either drop or fill. 

```python
X.isna().sum(axis=0).sum(axis=0)
```
0

## *Visualization*

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import seaborn as sns
import numpy.linalg as LA
from scipy import stats
```
### *Violin Plot*
We look at each feature to see its distribution and relationship in respect to the class label. Before plotting we normalize our data to be able to visualize all features in one plot and compare them together.

```python
scaler = StandardScaler()
z_fit = scaler.fit_transform(X)
Z=pd.DataFrame(data=z_fit, columns=X.columns)
Z_join = pd.concat([y, Z.iloc[:,:]], axis=1)
data = pd.melt(Z_join,id_vars="status", var_name="features", value_name='value')
data
```
<img src= "/assets/img/post1_melt.png">

```python
plt.figure(figsize=(26,10))
sns.violinplot(x="features", y="value", hue="status", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
```
<img src= "/assets/img/post1_violin.png">

For example PRDE, spread1 and PPE show a separation of their median value based on the class label. Therefore those variables are deemed important for "status" classification. On the other hand, NHR shows the same median for both classes. So it doesn't seem to be direcetly related to the class label.

### *Box plot*
Alternatively we can create box plot to get quartiles information for each feature.

```python
plt.figure(figsize=(26,10))
sns.boxplot(x="features", y="value", hue="status", data=data)
plt.xticks(rotation=90)
```

<img src= "/assets/img/post1_boxplot.png">

From the above box plot we observe that MDVP features have similar range of values and distribution. Provided they are corrolated we can drop the redundant features.

### *swarmplot*
swarmplot shows all the observations in the same context as violin or boxplot. 
We observe spread1 and PPE seem to be good features to separate the dataset based on the class label. On the contrary, class labels are spread across DFA, and it does not seem to be a good candidate to separate the data based on the class label.

```python
plt.figure(figsize=(20,10))
sns.swarmplot(x="features", y="value", hue="status", data=data)
plt.xticks(rotation=90)
```

<img src= "/assets/img/post1_swarmplot.png">

### *joinplot*
Moving to statistical analysis we are going to verify some of the observations that we made earlier. Below we see a correlation factor of 0.99 between two of the MDVP features, confirming a strong linear correlation. This is a redundancy in the dataset that can be removed.

```python
def plot_join_plot(df, feature, target):
  j = sns.jointplot(feature, target, data = df, kind ='reg')
  j.annotate(stats.pearsonr)
  return
  
plot_join_plot(X, 'MDVP:Shimmer', 'MDVP:Shimmer(dB)')
plt.show()
```
<img src= "/assets/img/post1_joinplot.png">

### *pairplot*
To evaluate the correlation of multiple variable at the same time we use PaiGrid from seaborn library.

```python
sns.set(style="white")
df1 = X.loc[:,['MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5']]
pp = sns.PairGrid(df1, diag_sharey=False)
# pp = sns.PairGrid(df1)

# fig = pp.fig 
# fig.subplots_adjust(top=0.93, wspace=0.3)
# t = fig.suptitle('corrolated features', fontsize=14)
# plt.show()
pp.map_upper(sns.kdeplot, cmap="cividis")
pp.map_lower(plt.scatter)
pp.map_diag(sns.kdeplot, lw=3)
```

<img src= "/assets/img/post1_pairgrid.png">

### *Heatmap*
We use heatmap to see the correlation map of the entire features.
Below are the features that are linearly correlated.
* MDVP:Jitter(%), MDVP:RAP, MDVP:PPQ, Jitter:DDP (we keep MDVP:PPQ)
* MDVP:Shimmer, MDVP:Shimmer(dB), 'Shimmer:APQ3', 'Shimmer:APQ5','MDVP:APQ', 'Shimmer:DDA' (we keep 'MDVP:APQ')
* spread1, PPE (we keep PPE)

We can look at the previous plots to decide which one of them we want to keep, and drop the rest.

```python
fig,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
```

<img src= "/assets/img/post1_heatmap1.png">

## *Drop linearly correlated columns*
Instead of visual inspection we could have directly extract the correlation coefficients from the correlation matrix.

```python
corr_pct = 0.98
col_corr = set()
for i in range(len(X.corr().columns)):
    for j in range(i):
        if abs(X.corr().iloc[i, j]) > corr_pct:
          print(f'{X.corr().columns[i]} and {X.corr().columns[j]} correlated by {X.corr().iloc[i, j]}')
          col = X.corr().columns[i]
          col_corr.add(col)
print(col_corr, len(col_corr), type(col_corr))
```
<img src= "/assets/img/post1_corr_columns.png">

```python
# Scatter Plot with Hue for visualizing data in 3-D
X_join = pd.concat([X[col_corr],y], axis=1)

pp = sns.pairplot(X_join, hue="status", size=1.8, aspect=1.8, 
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Most corrolated features', fontsize=14)
plt.show()
```

<img src= "/assets/img/post1_pairplot.png">

After dropping the redundant columns, The number of columns are reduced to 13.

```python
col_drop = ['MDVP:Jitter(%)', 'MDVP:RAP', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:DDA', 'spread1']
X = X.drop(col_drop, axis=1)
X.shape
```
(195, 13)

## Merge cleaned up data
* In case some rows from X were removed in the process, we would need to align rows od X and y again. We use inner merge between X and y to achieve it.

```python
df = pd.merge(X, y, left_index=True, right_index=True)
X = df[df.columns[:-1]]
y = df[df.columns[-1]]
```

## PCA Dimension Reduction
Now let's attempt to reduce the dimension of the cleaned up data by Principal Component Analysis technique.

Before fitting our data into PCA, it's standardized by StandardScalar utility class.

```pyhton
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

scaler = StandardScaler()
z_fit = scaler.fit_transform(X.values)
Z = pd.DataFrame(z_fit, index=X.index, columns=X.columns)

pca = PCA()
pca_features = pca.fit_transform(Z.values)
```

After fitting data into PCA model, we create the Scree plot. Looking at the cumulative sum of explained variance we need to keep only 5 component to retain 90% collective variance of our data.

```python
X_var_ratio = X.var()/(X.var().sum())

fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(18,5))
ax1.plot(range(len(X.var())), X_var_ratio.cumsum(), label="dataset features")
ax1.plot(range(len(X.var())), np.cumsum(pca.explained_variance_ratio_), label="PCA features")

ax1.set_title("Cumulative Sum of Explained Variance Ratio")
ax1.set_xlabel('PCA features')
ax1.set_ylabel('Total Explained Variance Ratio')
ax1.axis('tight')
ax1.legend(loc='lower right')

ax2.bar(x=range(len(X.columns)), height=pca.explained_variance_ratio_)

ax2.set_title('Scree Plot')
ax2.set_xlabel('PCA features')
ax2.set_ylabel('Explained Variance Ratio')

plt.show()
```

<img src= "/assets/img/post1_pca.png">

```python
pca_num = (np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9))[0][0]
pca = PCA(pca_num)
pca_features = pca.fit_transform(Z.values)
print(pca_features.shape, type(pca_features))
```
(195, 5) <class 'numpy.ndarray'>

```python
df_pca = pd.DataFrame(pca_features)
df_pca.head(2)
```
<img src= "/assets/img/post1_pcatab.png">

By looking at the swarmplot, we notice that principal components P0 and P3 are most related to our class label.

```python
pca_data = pd.melt(pd.concat([df_pca,y], axis=1),id_vars="status", var_name="PCA_features", value_name='value')
plt.figure(figsize=(10,8))
sns.swarmplot(x="PCA_features", y="value", hue="status", data=pca_data)
plt.xticks(rotation=90)
```

<img src= "/assets/img/post1_swarm2.png">

A scatter plot shows class label separation between P0 and P3.

```python
x_data = df_pca.iloc[:,0]
y_data = df_pca.iloc[:,3]
plt.figure(figsize=(10,8))
sns.scatterplot(
    x= x_data, y= y_data,
    hue=y,
    legend="full",
    alpha=0.8
)
plt.xlim(x_data.min(),0.8*x_data.max())
plt.ylim(y_data.min(),0.8*y_data.max())
plt.xlabel("P0")
plt.ylabel("P3")

plt.show()
```

<img src= "/assets/img/post1_pca_scatter_plot.png">

## K-means Clustering
Finally we utilize PCA features to cluster our data for classification. Since our class label is binary, we partition our data into two clusters.

```python
y.value_counts()
```
1    147
0     48
Name: status, dtype: int64

```python
sqr_err = []
for i in range(1,10):
  kmeans = KMeans(i)
  # kmeans.fit(df_pca.values)
  kmeans.fit(X)

  sqr_err.append(kmeans.inertia_)
  
plt.figure(figsize=(8,5))
plt.plot(range(1,10), sqr_err)
plt.xlabel("Number of clusters")
plt.ylabel("sum of square errors")
plt.title("# of Clusters for\nUnsupervised Learning")
plt.show()
```

<img src= "/assets/img/post1_kmeans1.png">

```python
kmeans=KMeans(2)
kmeans.fit(X)
```

To show the benefit of PCA, first we run the k-means cluster on original features and then the pca features. Then we compare their class separation by creating scatter plot.

```python
def scatter_comp(xdata, ydata, y, cluster_label):
  x_data = xdata
  y_data = ydata
  plt.figure(figsize=(16,10))
  
  ax0 = plt.subplot(2,2,1)
  sns.scatterplot(
    x= x_data, y= y_data,
    hue = y,
    cmap='viridis',
    legend="full",
    alpha=1,
    ax=ax0
    )
  ax0.set_xlim(x_data.min(),x_data.max())
  ax0.set_ylim(y_data.min(),y_data.max())
  ax0.set_xlabel(x_data.name)
  ax0.set_ylabel(y_data.name)
  ax0.set_title("Class label")

  ax1 = plt.subplot(2,2,2)
  sns.scatterplot(
    x= x_data, y= y_data,
    hue= cluster_label,
    cmap='viridis',
    legend="full",
    alpha=1,
    ax=ax1
    )
  ax1.set_xlim(x_data.min(),x_data.max())
  ax1.set_ylim(y_data.min(),y_data.max())
  ax1.set_xlabel(x_data.name)
  ax1.set_ylabel(y_data.name)
  ax1.set_title("Cluster Label")

  return plt.show()
  
scatter_comp(X["DFA"], X["MDVP:Fo(Hz)"], y, cluster_label)
```

<img src= "/assets/img/post1_scatter plot_kmeans1.png">

The above plot shows k-means clustering cannot effectively separate observations based on original data features. Next step, we will repeat the sam process on pca features.

```python
kmeans = KMeans(2)
kmeans.fit(pca_features)
```

scatter_comp(df_pca[0], df_pca[1], y, pca_label)
scatter_comp(df_pca[0], df_pca[2], y, pca_label)
scatter_comp(df_pca[0], df_pca[3], y, pca_label)

<img src= "/assets/img/post1_scatter_pca1.png">
<img src= "/assets/img/post1_scatter_pca2.png">
<img src= "/assets/img/post1_scatter_pca3.png">


```python
(y==pca_label).sum()/len(y)
```
0.6051282051282051

A larger training dataset and test dataset would give a better view of our classification performance. Nevertheless k-means clustering over pca features shows noticable classification improvement over original features.

## Conclusion
The process of inspecting, visualizing, cleaning, transforming, and modeling of the data with the objective of extracting useful information and drawing conclusion is data analysis. 
We took a numerical dataset related to Parkinson's disease provided by University of Oxford. We went through every steps of the above, to analyze this dataset, and show how we could extract features that could be used for classification.


