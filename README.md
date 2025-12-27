# COVID-on-Peoples-Daily-Life

This project analyzes a survey dataset capturing public opinions and behavioral changes during the 2020 COVID-19 lockdown. It includes exploratory data analysis (EDA), visualizations, and a logistic regression model to predict whether respondents prefer working/studying from home.

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run
1. Create and activate a virtual environment (recommended)
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
Run the script:

bash
Copy code
python main.py
What’s in the Analysis
The script generates visualizations that explore:

Average working hours before vs. during the pandemic by occupation

Work location preferences by gender and occupation

Work preference by age group

Demand for hybrid (WFH on certain days) by age

Total time (work + travel) before vs. during the pandemic by occupation

Home environment vs. productivity change

Logistic Regression: Predicting WFH Preference
In addition to EDA, the project trains a logistic regression classifier to predict whether a respondent prefers working/studying from home (prefer_wfh = 1) versus not (prefer_wfh = 0). Categorical variables are one-hot encoded and numeric features are standardized.

Model Performance (held-out test set):

Dataset used for modeling: n = 479 responses | 16 raw features

Accuracy: 0.925

F1 score: 0.870

ROC-AUC: 0.915

Confusion Matrix:

TN = 81, FP = 3

FN = 6, TP = 30

Classification Report (summary):

Class 0 (Not WFH): Precision 0.93, Recall 0.96, F1 0.95

Class 1 (WFH): Precision 0.91, Recall 0.83, F1 0.87

Top Drivers (largest |coefficients|):

certaindays_hw_Yes (+1.74)

line_of_work_Architect (+1.48)

age_33-40 (−1.18)

certaindays_hw_No (−1.16)

travel_time (−0.75)

Note: Coefficients reflect association with WFH preference, not necessarily causation.

Conclusion
Impact on Working Hours by Occupation: The comparison of working hours before and during the pandemic across occupations shows that for most occupations, working hours remained relatively stable or slightly decreased. However, the overall decrease in working hours suggests that the pandemic led to a shift in work-life balance or a change in workload for some professions.

Work Location Preferences by Gender: The graph comparing work location preferences by gender indicates that both men and women have a significant preference for working from home. However, a slightly higher proportion of women prefer remote work compared to men. The option of "prefer not to say" gender also highlights diversity in the dataset.

Work Location Preferences by Occupation: Across different occupations, there is a noticeable preference for working from home. Some occupations (such as those likely to be more flexible or tech-oriented) show a stronger preference for remote work, while others may still prefer in-person environments, reflecting industry-specific trends.

Work Preferences by Age: Age plays a role in determining work preferences, with younger individuals showing a stronger preference for remote work, while older individuals are more inclined to prefer in-person work or a hybrid approach. This suggests generational differences in adapting to remote work environments.

Work-from-Home Flexibility by Age: The need for working from home on certain days shows a strong association with age. Younger workers seem more adaptable to hybrid or flexible work models, while older workers may either prefer full remote work or full in-person attendance. Flexibility remains key for many individuals across age groups.

Total Working and Travel Hours by Occupation: There is a reduction in total working and travel hours during the pandemic across all occupations. The reduction in commuting time is likely a major factor, as many transitioned to remote work, making the workday more efficient but also potentially less structured.

Home Environment's Impact on Productivity: A positive home environment (comfortable, quiet, and conducive to work) appears to enhance productivity during remote work. Conversely, poor home environments can hinder productivity, showing the importance of workspace quality for employees who work from home.

In conclusion, the pandemic has influenced working patterns, with shifts towards remote work, reduced commuting, and changes in productivity based on home environments. Age, gender, and occupation all play roles in determining how individuals have adapted to these changes.