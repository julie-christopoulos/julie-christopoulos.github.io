# Machine Learning for Predicting Aerosol Mixed Layer Heights 



***

# Introduction 

The planetary boundary (PBL) is the lowest, turbulent layer of the atmosphere and serves to facilitate a multitude of feedbacks in the atmosphere, including those between the atmosphere, ocean, and land. The height of the boundary layer (PBLH) is responsible for governing many tropospheric activities, such as aerosol distributions, convection, and cloud formation [7]. Its complex evolution challenges observations of the PBL. NASA Langley (LaRC) hosts a suite of airborne lidar High Spectral Resolution Lidars (HSRL) and differential absorption lidars (DIAL) whose observables of aerosol properties (532nm backscatter) aid in identifying the PBLH. 

This project aims to improve mixed layer height (MLH) estimates derived from airborne HSRLs to allow for more automated retrievals over a wide range of atmospheric and surface conditions. Currently, the LaRC airborne lidar algorithm utilizes the Haar wavelet covariance transform method [3] to derive MLHs. Essentially, a Haar wavelet function is used to transform a lidar backscatter profile. The lowest altitude minimum of the transform is determined as the height of the boundary layer. The algorithm’s implementation on HSRL datasets is mostly automized, although it requires manual inputs of threshold values for WCT peak detection. In addition, a final manual quality control phase is necessary to correct remaining outliers by utilizing other observations or based on intuitive situations [9]. It is thus evident, further research is needed to create a more robust, accurate, and automated MLH algorithm that can produce reliable MLHs without user input. 

To address this, a supervised machine learning approach is taken using the ensemble learning method. Two lidar field campaigns are selected (CPEX-AW (2021), ACT-America (2019) to predict MLHs for several test flights. The resulting predictions are evaluated against the quality checked MLHs and the default method (using a single WCT threshold) to assess improvement. Overall, increased model performance with the inclusion of ensemble learning was observed compared to the default method of prediction, illustrating the advantages of automizing MLH predictions. Further work is needed to increase prediction accuracy and assess more complex lidar scenes. 



# Data

## Background

This project utilizes field campaign datasets obtained from the LaRC High Altitude Observatory Instrument (HALO) which employs the HSRL and DIAL techniques to provide profiles of aerosols and water vapor. Data is taken from two NASA field campaigns, ACT-America and CPEX-AW. 

**ACT-America 2019** (Atmospheric Carbon and Transport - America):
* <u>Location:<u> Eastern U.S. 
* <u>Mission Background:<u> Airborne campaign to study the transport and fluxes of atmospheric carbon dioxide and methane
* Instrument: HALO (High Altitude Lidar Observatory)

**CPEX-AW 2021** (Convective Processes Experiment - Aerosols and Winds):
* Location: St. Croix 
* Mission Background: Airborne campaign to study dynamics and microphysics related to the Saharan Air Layer, African Easterly Waves and Jets, Tropical Easterly Jet, and deep convection in the ITCZ. 
* Instrument: HALO (High Altitude Lidar Observatory)

## Predictor Selection

Tablle 1. illustrates the predictors selected for the ensemble learning algorithm. The first four predictions represent various MLHs derived using constant thresholds, ranging from 0.00001-0.01. These thresholds fall in the range of the typical thresholds selected in the current MLH algorithm, and are used to test the sensisity of particular threshold values. Predictors 5-9 are associated with these four height predictors. The variance of the 532nm aerosol backscatter gradient is computed 360m above and below the MLHs associated with the first four predictors. Predictors 10-13 are also associated with the first four height predictors. These predicors were computed to be the horizontal variance in MLHs. A temporal variance is taken 10 time steps before and after each of the derived heights (data is available every 10s). Next, the solar hour angle is computed from the geographic variables (i.e. latitude, longitude). This was selected as a predictor since the majority of the ACT-America dataset was collected during solar noon. Lastly, I computed a terrain flag based on the geographical coordinates of the observations. The terrain flag is incorporated since the dataset incoporates both marine- and terrestrial-type MLHs. 


| No. | Description                                                 |
| --- | ----------------------------------------------------------- |
| 1)  | MLH (Thresh = 0.00001)                                     |
| 2)  | MLH (Thresh = 0.0001)                                      |
| 3)  | MLH (Thresh = 0.001)                                       |
| 4)  | MLH (Thresh = 0.01)                                        |
| 5-9)| Vertical Variance in 532nm Backscatter Gradient (360m above and below MLHs)|
| 10-13) | Temporal Variance in MLHs associated with predictors 1-4  |
| 14) | Solar Hour Angle                                           |
| 15) | Terrain Flag (land = 0; water = 1)                         |

*Table 1: Predictors incorporated in the ensemble learning MLH algorithm.*

## Reference Data

The reference data for the ensemble learning algorithm is comprised of the quality-checked/manually adjusted MLHs. An example of what these MLHs look like for a flight is shown in Fig. 1. 


![gifbug](assets/IMG/20190724_F1_MLH.png)

*Figure 1: Quality-Checked MLHs shown for a flight on June 24, 2019 during the ACT-America campaign.*

# Modelling


Before implementing the ensemble model, all of the data within the training and testing datasets was standardized using the minimums and maximums (see code snippet below). The testing dataset was stadardized with the mins and maxs from the training set to elimate the possibility of bias. 

## Data Standardization

```matlab
%-- Find Minimums and Maximums -- 
[min_tr, max_tr] = cellfun(@find_range, predic_all_tr, 'UniformOutput', false);
[min_p1, min_p2, min_p3, min_p4,min_vvar_p1, min_vvar_p2, min_vvar_p3, min_vvar_p4, min_hvar_p1, min_hvar_p2, min_hvar_p3, min_hvar_p4, min_hangle,min_flag] = min_tr{:};

[max_p1, max_p2, max_p3, max_p4,max_vvar_p1, max_vvar_p2, max_vvar_p3, max_vvar_p4,max_hvar_p1, max_hvar_p2, max_hvar_p3, max_hvar_p4, max_hangle,max_flag] = max_tr{:};

%Archived Mins and Maxs
[min_arch, max_arch] = find_range(arch_tr);

%-- Standardize Values for Machine Learning --
std_values_tr = cellfun(@(x, min_val, max_val) standardizeData(x, min_val, max_val), predic_all_tr, min_tr, max_tr, 'UniformOutput', false);
[sp1_tr, sp2_tr, sp3_tr, sp4_tr,svvar_p1_tr, svvar_p2_tr, svvar_p3_tr, svvar_p4_tr,shvar_p1_tr, shvar_p2_tr, shvar_p3_tr, shvvar_p4_tr,shangle_tr,sflag_tr] = std_values_tr{:};
[sarch_tr] = standardizeData(arch_tr,min_arch,max_arch);
```

A **supervised learning** learning approach was implemented to predict altitudes of mixed layer heights (in meters) for multiple flights within the ACT-America and CPEX-CV field campaigns. 23 flights were randomized and split into training (19 flights) and testing (5 flights). A regression ensemble approach was implemented to model the height values. Bootstrap aggregating (bagging) was selected to help reduce overfitting and the impact of outliers within the training data. Additionally, I opted to enable the surrogate decision splits. I opted for this method since a number of the predictros contained missing data. This way if a predictor was missing for a particular observation, a decision could be made based on other available predictors. Lastly, for each tree in the ensemble, the MaxNumSplits was set to 200 in order to restrict the tree depth and control its complexity. After several test cases, I found this value performs well. 

## Implementation
```matlab
treeTemplate = templateTree('Surrogate','on','MaxNumSplits',200);
ensemble = fitrensemble(train_data,train_arch,'Method','Bag','Learners',treeTemplate);
save('ensemble_model.mat','ensemble');

mlh_ens = cell(length(sp1_tst), 1);
%Make Predictions for each flight
for i = 1:length(testFiles)
    test_MLH = sp1_tst{i};
    nan_index = find(isnan(test_MLH));
    test_data = [sp1_tst{i}', sp2_tst{i}', sp3_tst{i}', sp4_tst{i}',svvar_p1_tst{i}', svvar_p2_tst{i}', svvar_p3_tst{i}', svvar_p4_tst{i}',shvar_p1_tst{i}', shvar_p2_tst{i}', shvar_p3_tst{i}', shvar_p4_tst{i}',shangle_tst{i}',sflag_tst{i}'];
    %Ensemble Prediction + recover units
    ens_dat = predict(ensemble,test_data).* (max_arch - min_arch) + min_arch;
    ens_dat(nan_index) = nan;
    mlh_ens{i} = ens_dat';
end
```

# Results

## Predictor Importance 

![gifbug](assets/IMG/importance.png)

*Figure 2: Predictor importance for ensemble learning method.*

## Summary Statistics 

| Method                | NMB   | NME   | MB     | ME    | RMSE   | CORR |
|-----------------------|-------|-------|--------|-------|--------|------|
| Default               | -0.04 | 0.19  | -41.31 | 193.21| 324.94 | 0.81 |
| Ensemble Learning     | -0.01 | 0.09  | -14.14 | 92.94 | 157.39 | 0.96 |

*Table 2: Summary of Ensemble and default model performance.*

To assess the performance of the ensemble learning model, predicted values were evalated against the quality-checked MLHs (observed values). In addition, the performance of the default method was assessed for comparison. 

## Summary Plots

![gifbug](assets/IMG/scatter.png)

*Figure 4: Summary of the observed vs. predicted values for 5 test flights.*

![gifbug](assets/IMG/residuals.png)

*Figure 5: Summary of the residuals for 5 test flights.*


## Test Flight Results

### 20190710 ACT-America Test Flight 

![gifbug](assets/IMG/20190710_bsc.png)

*Figure 6: Predicted MLHs for ensemble (magenta) and quality-checked (white) methods for June 10, 2019 (ACT-America).*

| Method              | NMB   | NME   | MB      | ME     | RMSE   | CORR |
|---------------------|-------|-------|---------|--------|--------|------|
| Ensemble Learning   | -0.02 | 0.04  | -30.27  | 69.42  | 92.69  | 0.94 |


### 20210828 CPEX-AW Test Flight 
![gifbug](assets/IMG/20210828_bsc.png)

*Figure 7: Predicted MLHs for ensemble (magenta) and quality-checked (white) methods for August 28, 2021 (CPEX-AW).*

| Method            | NMB   | NME   | MB    | ME    | RMSE   | CORR |
|-------------------|-------|-------|-------|-------|--------|------|
| Ensemble Learning | 0.02  | 0.06  | 12.07 | 34.92 | 58.42  | 0.89 |

# Discussion



# Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

## Future Work
Future work on this topic would benefit from incorporating additional lidar field campaign data (e.g., LISTOS, ACTIVATE, TRACER-AQ, CPEX-CV, CAMP2Ex), lidar observables (1064mm aerosol depolarization, relative humidity, water vapor mixing ratio, aerosol typing), and PBLHs derived from different methods (PBLH derived from dropsonde data).


# References
[1] DALL-E 3

[back](./)

