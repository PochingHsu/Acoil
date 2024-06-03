# CFD Suggorate Models Development for A-coil Velocity Profile
This project uses ML models (e.g. ANN, KNN, GaussianProcess, RandomForest, DecisionTree and LinearRegression) to develop computionally efficient suggorate model with CFD simulation data.
The suggorate model can be used for the design optimization of A-coil or HVAC system.

## Generate CFD data

Firstly, we solved CFD model parametrically and output data for data analysis by using ANSYSFLEUNT-Python API: Pyfluent. The data processing flow is shown in below:

(File: PyFluent.py)

![PyFluent](https://github.com/PochingHsu/AcoilCFDSuggorate/assets/165426535/c9bb1667-af0b-444b-b057-d5c155792009)

## ML suggorate Models Development
The air velocity profile prediction of ANN model:

(File: DataSplit.py -> ann.py -> ann_pred.py)
 
![ANN](https://github.com/PochingHsu/AcoilCFDSuggorate/assets/165426535/d29352f2-9641-499e-ba8a-2d039441249e)

The air velocity profile prediction of GaussianProcess model:

(File: DataSplit.py -> ml.py -> ml_pred.py)
 
![GP](https://github.com/PochingHsu/AcoilCFDSuggorate/assets/165426535/a354ce75-4f19-48ed-a8eb-a78cbe580558)
