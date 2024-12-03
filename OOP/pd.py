
import pandas as pd

def last_four(num):
    return str(num)[-4:]

def yelp(price):
    if price<10:
        return '$'
    elif price >=10 and price < 30:
        return '$$'
    else:
        return '$$$'
df2 = pd.read_csv('tips.csv')
print(df2['CC Number'][0])
print(last_four(3560325168603410))
df2['last_four'] = df2['CC Number'].apply(last_four)
df2['Expensive'] = df2['total_bill'].apply(yelp)
print(df2['Expensive'])
print(df2['total_bill'].between(10,20,inclusive='both')) #prints total bill col with values between 10,20, both bounds inclusive
print(df2['price_per_person'].between(5,8,inclusive = 'both'))
print(df2.nlargest(10,'tip'))
print(df2.nsmallest(10,'tip'))






##############



#playing around with dataframes using pandas
import pandas as pd
df = pd.read_csv("heart.csv")
df['sex'] = df['sex'].replace({1:'male', 0:'female'})
print(df['sex'].head(10))
print(df.columns)
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
              'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels',
               'thalassemia', 'target']
df['chest_pain_type'] = df['chest_pain_type'].replace({0:'typical angina', 1:'atypical angina', 2:'non-anginal pain', 3:'asymptomatic'})
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].replace({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})
df['rest_ecg'] = df['rest_ecg'].replace({0:'normal', 1:'ST-T wave abnormality', 2:'left ventricular hypertrophy'})
df['exercise_induced_angina'] = df['exercise_induced_angina'].replace({0:'no', 1:'yes'})
df['st_slope'] = df['st_slope'].replace({1:'upsloping', 2:'flat', 3:'downsloping'})
df['thalassemia'] = df['thalassemia'].replace({1:'normal', 2:'fixed defect', 3:'reversable defect'})
print(df.head(10))
df = df.rename(columns = {'rest_ecg':'ecg', 'sex':'gender'}) #for renaming the columns
df = df.rename(columns = {'age':"AGE"})
df = df.rename(columns = {'chest_pain_type':'Chest pain type'})
print(df.head())
print(df.describe())#returns info about data frame
#inserting columns
df = pd.read_csv("heart.csv") #reloading original file
df = df.head(10) #we resize the data frame by init it to first 10 rows since the list we use for inserting the names will be of size 10
Names = ['Ali', 'Salman', 'Sohail', 'Mohsin', 'Waqas', 'Zeshan', 'Babar', 'John', 'Elon', 'Michael']
df.insert(0,'name', Names) #(position(loc), column name, column data(a list)
print(df.head())
df = df.head() #resize to first 5 rows
df = df[['age', 'sex', 'cp', 'target']] #now the data frame only has columns mentioned
print(df)

#dropping cols
target_data = df['target'] #init target col to a target_data list
df.drop(labels = ['target'], axis=1, inplace=True) #inplace = True changes the original dataframe while inplace=False makes the copy
print(df)
df.insert(3, 'target', target_data) #inserting target back into the data frame
print(df)
cp_data = df['cp']
df_copy = df.drop(labels = ['cp'], axis = 1, inplace = False)
print(df) #since inplace=False when df.drop() was used, df will still have cp col
print(df_copy) #df_copy will not have the cp col
df.drop(labels = ['cp'], axis = 1, inplace = True) #the cp col will now be removed from the og df since inplace = True
print(df)
df.insert(2,'cp', cp_data) #inserting cp back into the og df
print(df)

#adding rows
new_row = {'age':25, 'sex':1, 'cp':3, 'target':0}
df.loc[df.index.size - 1] = new_row #df.index.size = 5, so new_row will be added at index#4
#ignore_index is False by default, if True, index values are not used along concatenation axis
#and the resulting axis will be labeled 0,1....,n-1, can't append dictionary if ignore_index = False
print(df)

#adding rows at a specific index
df.loc[2] = [49, 0, 2, 0] #the existing row at index 2 will be replaced
print(df)

#dropping rows
df = df.drop(labels = [2,3], axis = 'rows') #the df will now have indexes 0,1,4
print(df)
new_row = [{'age':49, 'sex':0, 'cp':2, 'target':0}, {'age':61, 'sex':1, 'cp':0, 'target':0}]
df.loc[df.index.size - 1] =new_row[0] #df.index.size = 3, so new_row will be added as index#2
print(df) #df now has indexes 0,1,4,2

#to reset indexes from 0-4, we will use reset_index
df.reset_index(drop = True, inplace=True)
print(df) #indexes are now 0,1,2,3








##############################



import pandas as pd
#concatenating two data frames
df = pd.read_csv('heart.csv').head()
df2 = pd.read_csv('heart.csv').tail()
result = pd.concat([df,df2]) #concatenating rows
print(result) #this will not reset indexes
result = pd.concat([df,df2], ignore_index=True) #ignore_index = True allows us to reset indexes when we concat dataframes
print(result)
df=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',]]
df2=df2[['slope','ca','thal','target']]
result = pd.concat([df,df2], axis = "columns") #usually rows concat will have ignore_index set to true because ideally we would want to reset indexes but for columns we don't want to change the labels
print(result)
result.reset_index(drop = True, inplace=True)
print(result)
result = pd.concat([df,df2], axis = "columns", ignore_index=True) #this will change column labels to numberings starting from 0
print(result)
print(df[['age']]) #extraction of particular col
print(df[['age', 'sex']]) #extraction of multiple cols
print(df)
print(df.loc[1][3]) #iloc[][] allows access to rows and columns by indexes (might give Future Warning since method is modified in pandas 3.0)
print(df.loc[1]['trestbps']) #loc[][] allows access to rows and columns by labels
print(df.loc[df['age']>60]) #df[df['age']>60] prints the same thing
names = ['A','B','C','D','E']
df.insert(2,'name',names)
print(df)
df = df.set_index('name') #sets the name columns as the index
print(df)
df.sort_values('age', inplace=True)
print(df)
print(df.at['A', 'age']) #returns value at that pos
df.at['A', 'age'] = 100
print(df)

#derived columns can be made
df['new col'] = df['age'] + df['thalach'] #Col values are added for each particular position
print(df)















