---
layout: post
title: Step by Step Data Analysis and Visualization
#subtitle: Oxford Parkinson's Disease Detection Dataset
gh-repo: https://github.com/skhabiri/DS-Unit-1-Build
gh-badge: [star, fork, follow]
tags: [Data Analysis]
image: /assets/img/post1_cover3.png
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
<img src= "/assets/img/post1_dtype.png">


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



