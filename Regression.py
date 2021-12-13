data2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')
data2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')
data2017 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')
data2018 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')
data2019 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

#tidying the data
cols2015 =[0,1,2,4,11]
cols2016 = [0,1,2,4,5,12]
cols2017 = [0,1,3,4,11]
cols2018 = [0,1]
cols2019 = [0,1]

data2015.drop(data2015.columns[cols2015], axis=1, inplace=True)
data2016.drop(data2016.columns[cols2016], axis=1, inplace=True)
data2017.drop(data2017.columns[cols2017], axis=1, inplace=True)
data2018.drop(data2018.columns[cols2018], axis=1, inplace=True)
data2019.drop(data2019.columns[cols2019], axis=1, inplace=True)


columnNames = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom', 'Generosity','Trust (Government Corruption)']
data2015 = data2015.reindex(columns=columnNames)
data2016 = data2016.reindex(columns=columnNames)

mapping2015 = {data2015.columns[0]:'Score',data2015.columns[1]:'Economy', data2015.columns[2]: 'Family',data2015.columns[3]: 'Health',data2015.columns[4]: 'Freedom',data2015.columns[5]: 'Generosity',data2015.columns[6]: 'Trust'}
mapping2016 = {data2016.columns[0]:'Score',data2016.columns[1]:'Economy', data2016.columns[2]: 'Family',data2016.columns[3]: 'Health',data2016.columns[4]: 'Freedom',data2016.columns[5]: 'Generosity',data2016.columns[6]: 'Trust'}
mapping2017 = {data2017.columns[0]:'Score',data2017.columns[1]:'Economy', data2017.columns[2]: 'Family',data2017.columns[3]: 'Health',data2017.columns[4]: 'Freedom',data2017.columns[5]: 'Generosity',data2017.columns[6]: 'Trust'}
mapping2018 = {data2017.columns[0]:'Score',data2018.columns[1]:'Economy', data2018.columns[2]: 'Family',data2018.columns[3]: 'Health',data2018.columns[4]: 'Freedom',data2018.columns[5]: 'Generosity',data2018.columns[6]: 'Trust'}
mapping2019 = {data2019.columns[0]:'Score',data2019.columns[1]:'Economy', data2019.columns[2]: 'Family',data2019.columns[3]: 'Health',data2019.columns[4]: 'Freedom',data2019.columns[5]: 'Generosity',data2019.columns[6]: 'Trust'}


data2015 = data2015.rename(columns = mapping2015)
data2016 = data2016.rename(columns = mapping2016)
data2017 = data2017.rename(columns = mapping2017)
data2018 = data2018.rename(columns = mapping2018)
data2019 = (data2019.rename(columns = mapping2019)).dropna()


indiv = [data2015,data2016,data2017,data2018]
tempData = (pd.concat(indiv)).dropna()
TrainScores = tempData['Score']
TrainPredictors = tempData.drop('Score',1)
TestScores = data2019['Score']
TestPredictors = data2019.drop('Score',1)

#simple linear regression
LR = LinearRegression()
LR.fit(TrainPredictors,TrainScores)
y_TrainPred = LR.predict(TrainPredictors)
y_TestPred = LR.predict(TestPredictors)
print(mean_squared_error(TrainScores, y_TrainPred)) #On 2015-2018
print(mean_squared_error(TestScores, y_TestPred)) #On 2019

#ridge regression
clf = Ridge(alpha=0.1)
clf.fit(TrainPredictors,TrainScores)
y_TrainPred = clf.predict(TrainPredictors)
y_TestPred = clf.predict(TestPredictors)
print(mean_squared_error(TrainScores, y_TrainPred)) #On 2015-2018
print(mean_squared_error(TestScores, y_TestPred)) #On 2019 

#lasso regression
clf = linear_model.Lasso(alpha=0.1)
clf.fit(TrainPredictors,TrainScores)
y_TrainPred = clf.predict(TrainPredictors)
y_TestPred = clf.predict(TestPredictors)
print(mean_squared_error(TrainScores, y_TrainPred)) #On 2015-2018
print(mean_squared_error(TestScores, y_TestPred)) #On 2019 

plt.figure(figsize=(12,10))
cor = tempData.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()
