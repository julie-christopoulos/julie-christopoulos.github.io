## Machine Learning for Predicting Aerosol Mixed Layer Heights 

Machine learning technqiues were applied to predict the heights of the aerosol mixed layer (MLH) using an airborne lidar dataset.

***

## Introduction 

The planetary boundary layer height (PBLH) influences various troposheric processes including aerosol distributions, convection, and cloud formation. However, its complex evolution challenges observations in the PBL. Currently, NASA Langely employs airborne High Spectral resolution lidars (HSRL) and Differential Absorption Lidars (DIAL) to obtain vertical profiles of various atmospheric constituents at high spatial and temporal evolutions. These observables play a crucial role in the derivation of the aerosol mixed layer height (e.g. the height at which pariculates are well-mixed). Currently, MLHs are derived using an algorithm based on the Haar wavelet covariance transform method (WCT), which relies upon manually-adjusted threshold values for accurate MLH estiamtes. In addition, the estimations require a time-consuming quality-inspection process to correct remaining outliers. To create a more automated algorithm, this project utilizes datasets from HALO (High Spectral Resolution Lidar) and two field campaigns CPEX-AW (2021), ACT-America (2019) as inputs to a supervised machine learning algorithm. Since the predictions are height estimations, a regression-based algorithm is selected (ensemble method). 





We did this to solve the problem. We concluded that...

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

