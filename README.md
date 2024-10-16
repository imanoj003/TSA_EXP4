# Developer Name : Manoj M
# Reg no: 212221240027

# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 30 / 9 / 24



### AIM:
To implement ARMA model in python student performance datset.


### ALGORITHM:
```
1.Prepare the Data: Extract a time series from the dataset. In this case, we can use FinalGrade as our time series.
2.Check for Stationarity: Use statistical tests like the Augmented Dickey-Fuller test to check if the time series is stationary.
3.Fit the ARMA Model: Use the statsmodels library to fit the ARMA model to the data.
4.Plot the Results: Visualize the actual vs. predicted values.

```
### PROGRAM:
```
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller



 df = pd.read_csv('/content/student_performance.csv')



final_grades = df['FinalGrade']

result = adfuller(final_grades)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If p-value is > 0.05, the series is not stationary. You may need to difference the series.
if result[1] > 0.05:
    print("The series is not stationary. Differencing the series...")
    final_grades_diff = final_grades.diff().dropna()
else:
    final_grades_diff = final_grades


model = sm.tsa.ARMA(final_grades, order=(1, 1))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(final_grades, label='Actual Final Grades', marker='o')
plt.plot(model_fit.fittedvalues, color='red', label='Fitted values', marker='x')
plt.title('Actual vs Fitted Final Grades')
plt.xlabel('Student Index')
plt.ylabel('Final Grade')
plt.legend()
plt.show()
```

# OUTPUT:


![Untitled](https://github.com/user-attachments/assets/39c5de37-c48e-4037-a744-5e9e41a27470)


# RESULT:
   Thus a python program is created to fir ARMA Model successfully.
