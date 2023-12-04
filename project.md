## Machine Learning for Predicting Aerosol Mixed Layer Heights 

Machine learning technqiues were applied to predict the heights of the aerosol mixed layer (MLH) using an airborne lidar dataset.

***

## Introduction 

The planetary boundary layer height (PBLH) influences various troposheric processes including aerosol distributions, convection, and cloud formation. However, its complex evolution challenges observations in the PBL. Currently, NASA Langely employs airborne High Spectral resolution lidars (HSRL) and Differential Absorption Lidars (DIAL) to obtain vertical profiles of various atmospheric constituents at high spatial and temporal evolutions. These observables play a crucial role in the derivation of the aerosol mixed layer height (e.g. the height at which pariculates are well-mixed). Currently, MLHs are derived using an algorithm based on the Haar wavelet covariance transform method (WCT), which relies upon manually-adjusted threshold values for accurate MLH estiamtes. In addition, the estimations require a time-consuming quality-inspection process to correct remaining outliers. To create a more automated algorithm, this project utilizes datasets from HALO (High Spectral Resolution Lidar) and two field campaigns CPEX-AW (2021), ACT-America (2019) as inputs to a supervised machine learning algorithm. Since the predictions are height estimations, a regression-based algorithm was selected (i.e. regression ensemble model). 

We did this to compute mixed layer heights for five flights within ACT-America and CPEX-AW. We see increased model performance compared to the default method with the inclusion of ensemble learning methods, illustrating an advantage for automizing mixed layer height prediction. Further work is needed to address more complex scenes and environments for more accuarte prediction. 

## Data Background

The dataset is comprised of data from two field campaigns ACT-America and CPEX-AW. 

ACT-America (Atmospheric Carbon and Transport - America):
Location: Eastern U.S. 
Mission Background: Airborne campaign to study the transport and fluxes of atmospheric carbon dioxide and methane
Instrument: HALO (High Altitude Lidar Observatory)

CPEX-AW (Convective Processes Experiment - Aerosols and Winds):
Location: St. Croix 
Mission Background: Airborne campaign to study dynamics and microphysics related to the Saharan Air Layer, African Easterly Waves and Jets, Tropical Easterly Jet, and deep convection in the ITCZ. 
Instrument: HALO (High Altitude Lidar Observatory)

## Predictor Selection

Since the current method of MLH prediction relies upon specific thresholds, the predictors included are selected mainly to weigh the sensitivity of thresholds.

1) MLH (Thresh = 0.00001)
2) MLH (Thresh = 0.0001)
3) MLH (Thresh = 0.001)
4) MLH (Thresh = 0.01)
5-9) Vertical Variance in 532nm Backscatter Gradient (360m above and below MLHs)
10-13) Temporal Variance in MLHs associated with predictors 1-4
14) Solar Hour Angle
15) Terrain Flag (land = 0; water = 1)

## Reference Data

The quality-checked MLH data will serve as the observed dataset.

![](assets/variables.gif){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

A supervised learning approach was implemented to predict altitudes of mixed layer heights (in meters) for multiple flights within the ACT-America and CPEX-CV field campaigns. 23 flights were randomized and split into training (19 flights) and testing (5 flights). A regression ensemble approach was implemented to model the heights. 

```matlab 
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

