---
layout: post
title: Evaluation of Multi-class Classification Techniques on The Forest Cover Type Dataset
#subtitle: Forest Cover Type
gh-repo: https://github.com/skhabiri/DS17-Unit-2-Build
gh-badge: [star, fork, follow]
tags: [Predictive Modeling]
image: /assets/img/post2_pairplot2.png
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

data = pd.read_csv('https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/data/train.csv')
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

<img src= "../../../../DS17-Unit-2-Build/blob/master/figures/post2_heatmap.png">

Looking at scatter plot of different features, it seems "Elevation" is an important parameter in separating different classes of target label.


<p float="left">
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_pairplot1.png" width="450" />
      <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_pairplot2.png" width="450" /> 
</p>

We make the following observations from the Count plots below.
* Soil_Type10 shows a significant class distinction for Cover_Type=6.
* Similarly Wilderness_Area4 and Cover_Type=4 are strongly associated.

<img src= "https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_countplot1.png">

List of unique values in the feature shows Soil_Type15 and Soil_Type7 are constant. Additionally, Id column is a unique identifier for each observation. We'll drop those three columns as they do not carry any information to identify the target label.
There are also skewness in some features. "Horizontal_Distance_To_Hydrology" is an example of that.
<img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_skew.png">

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

### Baseline model
Normalized value counts of the target label shows an equal weight distribution for all classes.
```python 
y_train.value_counts(normalize=True)
```
7    0.142857
6    0.142857
5    0.142857
4    0.142857
3    0.142857
2    0.142857
1    0.142857
Name: Cover_Type, dtype: float64

### Pipeline Estimators, Feature_importances and permutation_importance
We are going to look at five different classifiers to compare their performance. For the regression classifiers we will standardize the data before fitting.

1. LogesticRegression
2. RidgeClassifier
3. RandomForestClassifier
4. GradientBoostingClassifier
5. XGBCLassifier

The following plots show the feature_importances and permutation importances of the tree-base classifiers. "Elevation" is given a higher importance weight compared to other features, as we saw earlier. Without the help of permutation, XGBClassifier does not seem to detect the importance of "Elevation" feature.

<img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_feature1.png">
<img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_feature2.png">
<img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2_feature3.png">

### XGBClassifier ovrerfitting
XGBClassifier tends to overfit and its growth parameter needs to be controlled. To see that effect, an early stop fitting with 50 rounds is run.
```python
# XGBoost early stop fit
xform = make_pipeline(
    FunctionTransformer(wrangle, validate=False), 
    # ce.OrdinalEncoder(),
)

xform.set_params(functiontransformer__kw_args = kwargs_dict)

X_train_xform = xform.fit_transform(X_train)
X_val_xform = xform.transform(X_val)

clf = XGBClassifier(
    n_estimators = 1000,
    max_depth=8,
    learning_rate=0.5,
    num_parallel_tree = 10,
    n_jobs=-1
)

#eval_set
eval_set = [(X_train_xform, y_train), (X_val_xform, y_val)]

clf.fit(X_train_xform, y_train, 
          eval_set=eval_set, 
          eval_metric=['merror', 'mlogloss'], 
          early_stopping_rounds=50,
          verbose=False) # Stop if the score hasn't improved in 50 rounds

print('Training Accuracy:', clf.score(X_train_xform, y_train))
print('Validation Accuracy:', clf.score(X_val_xform, y_val))
```
Training Accuracy: 0.9987599206349206
Validation Accuracy: 0.8528439153439153

XGBoost trains upto 0.999 and yields 0.85 validation accuracy. The following graph shows how validation error remains constant while training error reduces.

<img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-earlystop.png">

### Cross Validation Curve

The training set is split into kfolds and the model is cross validated based on a classifier parameter.

```python
kfold=4
for k, estimator in enumerate(tree_list):
  print(f"********** {est_dict[estimator][0].__class__.__name__} **********")
  classifier_name = clf_name(estimator, est_dict)  
  # estimator.set_params(functiontransformer__kw_args = kwargs_dict)
  # print(est_dict[estimator][1])
  
  param_distributions = {classifier_name.lower()+'__'+ est_dict[estimator][1][j]: est_dict[estimator][2][j] for j in range(len(est_dict[estimator][1]))}
  
  for i in range(len(est_dict[estimator][1])):
    param_name=classifier_name.lower()+'__'+ est_dict[estimator][1][i]
    param_range = est_dict[estimator][2][i]
    estimator.set_params(functiontransformer__kw_args = kwargs_dict)

    train_scores, val_scores = validation_curve(estimator, X_train, y_train,
    # param_name='functiontransformer__kw_args',
    param_name=param_name, 
    param_range=param_range, 
    scoring='accuracy', cv=kfold, n_jobs=-1, verbose=0
    )

    print(f'**** {param_name} ****\n')
    print("val scores mean:\n", np.mean(val_scores, axis=1))

    # Averaging CV scores
    fig, ax = plt.subplots(figsize=(12,8))

    ax.plot(param_range, np.mean(train_scores, axis=1), color='blue', label='mean training accuracy')
    ax.plot(param_range, np.mean(val_scores, axis=1), color='red', label='mean validation accuracy')
    ax.set_title(f'Cross Validation with {kfold:d} folds', fontsize=20)
    ax.set_xlabel(param_name, fontsize=18)
    ax.set_ylabel('model score: Accuracy', fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(which='both')
```

<p float="left">
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-rfc_maxdepth.png" width="300" />
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-gbc_maxdepth.png" width="300" /> 
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-xgbc_maxdepth.png" width="300" /> 
</p>

Gradient Boost classifier family quickly overfit at max_depth>6. So it's important to keep the tree depth shallow.

<p float="left">
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-rfc_sampleaf.png" width="300" />
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-rfc_maxfeat.png" width="300" />
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-gbc_maxfeat.png" width="300" /> 
</p>

max_feat parameter shows validation score saturates at numbers above 20, and starts to overfit. Lowering min_samples_leaf improves validation score. Hence we consider small numbers.

<p float="left">
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-xgb_learningrate.png" width="300" />
  <img src="https://github.com/skhabiri/DS17-Unit-2-Build/blob/master/figures/post2-xgb_childweight.png" width="300" />
</p>

For XGBoost we can slow down the overfitting issue with min_child_weight and learningrate parameters.

### Hyper Parameter Tunning
The parameters of all five estimators are optimized by RandomizedSearchCV.
```python
kfold=4
n_iter = 10
best_ests = [np.NaN]*len(est)
best_scores = [np.NaN]*len(est)
best_params = [np.NaN]*len(est)

for i, estimator in enumerate(est):
  print(f"\n********** {est_dict[estimator][0].__class__.__name__} **********")
  estimator.set_params(functiontransformer__kw_args = kwargs_dict)

  classifier_name = clf_name(estimator=estimator, est_dict=est_dict)
  print(est_dict[estimator][1])
  
  param_distributions = {classifier_name.lower()+'__'+ est_dict[estimator][1][j]: 
                         est_dict[estimator][2][j] for j in range(len(est_dict[estimator][1]))}

  rscv = RandomizedSearchCV(estimator, param_distributions=param_distributions, 
    n_iter=n_iter, cv=kfold, scoring='accuracy', verbose=1, return_train_score=True, 
    n_jobs=-1)
  
  rscv.fit(X_train, y_train)
  best_ests[i] = rscv.best_estimator_
  best_scores[i] = rscv.best_score_
  best_params[i] = rscv.best_params_

best_ypreds = [best_est.predict(X_val) for best_est in best_ests]
best_testscores = [accuracy_score(y_val, best_ypred) for best_ypred in best_ypreds]
print('Test accuracy\n', best_testscores)
print('best cross validation accuracy\n', best_scores)
```
Test accuracy
 [0.6911375661375662, 0.6326058201058201, 0.7754629629629629, 0.8700396825396826, 0.8498677248677249]
best cross validation accuracy
 [0.7080026455026454, 0.6332671957671958, 0.7709986772486773, 0.8645833333333334, 0.8445767195767195]

The accuracy scores above corresponds to 
LogisticRegression, RidgeClassifier RandomForestClassifier, GradientBoostingClassifier, XGBClassifier accordingly.
In this work, GradientBoostingClassifier with 87% accuracy on the test data shows the best performance among the five. 

























## Conclusion
The process of inspecting, visualizing, cleaning, transforming, and modeling of the data with the objective of extracting useful information and drawing conclusion is data analysis. 
We took a numerical dataset related to Parkinson's disease provided by University of Oxford. We went through every steps of the above, to analyze this dataset, and show how we could extract features that could be used for classification.

