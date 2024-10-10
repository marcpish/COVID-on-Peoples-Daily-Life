
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression


#I will be delving into peoples opinions on the lockdown in 2020 and will be looking to see any major points or correlation in certain data

data = pd.read_csv("psyco.csv")
data['age'] = data['age'].replace('Dec-18','0-18')

#Graph of Average working hours before and during the pandemic by occupation 
time_spent_graph = data[['occupation','time_bp','time_dp']]
time_spent_graph = time_spent_graph.groupby('occupation').mean()
time_spent_graph.plot(kind = 'bar', figsize =(10,6), color = ['blue','green'])
plt.xlabel('Occupation')
plt.ylabel('Average working hours')
plt.title('Average working hours before and during the pandemic by occupation')
plt.xticks(rotation=45)
plt.legend(['Time Spent on work before the pandemic', 'Time spent on work during the pandemic'])
plt.tight_layout()
#plt.show()

#Graph of work type preferred by gender
fig = plt.subplots(figsize=(9, 5))
sns.countplot(x='prefer', hue='gender', data=data, palette = ['blue','pink','red'])
plt.title('Preference for location of work by gender')
plt.xlabel('Prefernce to work from home or in person')
plt.ylabel('Person Count')
plt.legend(['Male','Female','Prefer not to say'])
#plt.show()

#Graph of work type preferred by occupation
fig2 = plt.subplots(figsize=(10, 5))
sns.countplot(x='prefer', hue='occupation', data=data, palette = 'viridis')
plt.title('Preference for location of work occupation')
plt.xlabel('Prefernce to work from home or in person')
plt.ylabel('Person Count')
plt.legend(title = 'Occupation')
plt.tight_layout()
plt.show()

#Graph of preferred work type by age
age_vs_work = data[['age', 'prefer']]

grouped_data = age_vs_work.groupby(['age', 'prefer']).size().unstack(fill_value=0)

grouped_data.plot(kind='bar', stacked=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Preferred Work Type by Age')
plt.legend(title='Preferred Work Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Graph of if working from home is needed with age
age_vs_workingfromhome = data[['age', 'certaindays_hw']]

grouped_data = age_vs_workingfromhome.groupby(['age', 'certaindays_hw']).size().unstack(fill_value=0)

grouped_data.plot(kind='bar', stacked=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('If Working From Home on Certain Days Is Needed by Age')
plt.legend(title='Work From Home Needed by Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Travel and work hours before and during pandemic by occupation
travel_work = data[['occupation', 'time_bp', 'time_dp', 'travel_time']]
travel_work['total_time_bp'] = travel_work['time_bp'] + travel_work['travel_time']
travel_work['total_time_dp'] = travel_work['time_dp'] + travel_work['travel_time']
travel_work = travel_work.groupby('occupation').mean()

travel_work[['total_time_bp', 'total_time_dp']].plot(kind='bar', figsize=(10,6), color=['blue', 'green'])
plt.xlabel('Occupation')
plt.ylabel('Total Hours (Work + Travel)')
plt.title('Total Working Hours (Work + Travel) Before and During Pandemic by Occupation')
plt.xticks(rotation=45)
plt.legend(['Before Pandemic', 'During Pandemic'])
plt.tight_layout()
plt.show()

# Home environment and productivity change
home_prod = data[['home_env', 'prod_inc']]
home_prod_grouped = home_prod.groupby(['home_env', 'prod_inc']).size().unstack(fill_value=0)
home_prod_grouped.plot(kind='bar', stacked=True)
plt.xlabel('Home Environment')
plt.ylabel('Count')
plt.title('Impact of Home Environment on Productivity')
plt.legend(title='Productivity Increase')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
