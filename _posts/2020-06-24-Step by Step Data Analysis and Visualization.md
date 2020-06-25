---
layout: post
title: Step by Step Data Analysis and Visualization
subtitle: Oxford Parkinson's Disease Detection Dataset
gh-repo: https://github.com/skhabiri/DS-Unit-1-Build
gh-badge: [star, fork, follow]
tags: [Data Analysis]
comments: true
---

# ***Data Wrangling And Visualization***
* **Dataset:** https://archive.ics.uci.edu/ml/datasets/Parkinsons

* **Dataset information:**
This dataset is composed of a range of biomedical voice measurements from people with Parkinson's disease (PD) or without PD. Each column in the table is a particular voice measure, and each row corresponds one of voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.
> 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

## *Loading dataset*


```import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data_set = pd.read_csv(url)```


