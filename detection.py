import pickle
import re
import string
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
import category_encoders as ce
from warnings import filterwarnings
from shapely.wkt import loads
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import math
import calendar
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import pyproj as _proj
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import xgboost as xgb
import lightgbm as lgb


main = tk.Tk()
main.title("Accident risk score prediction by using ML")
main.geometry("1300x700")

global filename
global df
global model
le = LabelEncoder()
le_features=[]
cat_features=[]
global train, test, sample_submission , population, roads_network
global x_train, X_test, y_train, y, x, train_corr
n_folds = 10


columns=['Police_Force','1st_Road_Class','1st_Road_Number','2nd_Road_Class','Speed_limit','Local_Authority_(District)','Local_Authority_(Highway)','Number_of_Vehicles','Urban_or_Rural_Area','Road_Type']
object_columns = ['Number_of_Vehicles','1st_Road_Class','Road_Type','Speed_limit','2nd_Road_Class','Pedestrian_Crossing-Human_Control','Pedestrian_Crossing-Physical_Facilities','Light_Conditions','Weather_Conditions','Road_Surface_Conditions','Special_Conditions_at_Site','Carriageway_Hazards','Did_Police_Officer_Attend_Scene_of_Accident','state','month_in_year','Urban_or_Rural_Area','dayofweek']
int_columns = ['Variable: All usual residents; measures: Value',
               'Variable: Males; measures: Value',
               'Variable: Females; measures: Value',
               'Variable: Lives in a household; measures: Value',
               'Variable: Lives in a communal establishment; measures: Value',
               'Variable: Schoolchild or full-time student aged 4 and over at their non term-time address; measures: Value',
               ]

float_columns = ['Variable: Area (Hectares); measures: Value',
                 'Variable: Density (number of persons per hectare); measures: Value',
                 'distance to the nearest point on rd',
                 'length',
                 'Latitude',
                 'Longitude',
                 'x',
                 'y']

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")

def reading_csv_files():
    global train, test, sample_submission , population, roads_network

    train = pd.read_csv(f'{filename}/train.csv')
    test  = pd.read_csv(f'{filename}/test.csv')
    sample_submission  = pd.read_csv(f'{filename}/sample_submission.csv')
    population = pd.read_csv(f'{filename}/population.csv')
    roads_network = pd.read_csv(f'{filename}/roads_network.csv')
    
def datasetCleaning():
    global filename
    global df
    global train, test, sample_submission , population, roads_network, x_train, X_test, y_train, y, x
    text.delete('1.0', END)
    reading_csv_files()
    text.insert(END, 'There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1])+"\n")
    text.insert(END, 'There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1])+"\n")
    text.insert(END, 'There are {} rows and {} columns in sample_submission'.format(sample_submission.shape[0],sample_submission.shape[1])+"\n")
    train["Time"]=train["Time"].fillna("00:00")
    train["Road_Surface_Conditions"]=train["Road_Surface_Conditions"].fillna(train["Road_Surface_Conditions"].mode()[0])
    train["Special_Conditions_at_Site"]=train["Special_Conditions_at_Site"].fillna(train["Special_Conditions_at_Site"].mode()[0])
    #test["Time"]=test["Time"].fillna(test["Time"].mode()[0])
    test["Time"]=test["Time"].fillna("00:00")
    test["Road_Surface_Conditions"]=test["Road_Surface_Conditions"].fillna(test["Road_Surface_Conditions"].mode()[0])
    test["Special_Conditions_at_Site"]=test["Special_Conditions_at_Site"].fillna(test["Special_Conditions_at_Site"].mode()[0])
    train['Date'] = train['Date'] + " " + train['Time']
    train['Date'] =  pd.to_datetime(train['Date'], format='%d/%m/%y %H:%M')

    test['Date'] = test['Date'] + " " + test['Time']
    test['Date'] =  pd.to_datetime(test['Date'], format='%d/%m/%y %H:%M')
    #train
    train["day_in_month"]=train.Date.dt.day
    train["month_in_year"]=train.Date.dt.month
    train["year"]=train.Date.dt.year
    train["hour"] = train["Date"].dt.hour
    train['dayofweek'] = train['Date'].dt.dayofweek
    train['Minute']=train['hour']*60.0+train["Date"].dt.minute
    train['WeekofYear'] = train['Date'].apply(lambda x : x.weekofyear)
    #test
    test["day_in_month"]=test.Date.dt.day
    test["month_in_year"]=test.Date.dt.month
    test["year"]=test.Date.dt.year
    test["hour"] = test["Date"].dt.hour
    test['dayofweek'] = test['Date'].dt.dayofweek
    test['Minute']=test['hour']*60.0+test["Date"].dt.minute
    test['WeekofYear'] = test['Date'].apply(lambda x : x.weekofyear)
    df1=test.copy()
    text.insert(END, 'There are {} rows and {} columns in trainafter preprocessing'.format(train.shape[0],train.shape[1])+"\n")
    text.insert(END, 'There are {} rows and {} columns in test after preprocessing'.format(test.shape[0],test.shape[1])+"\n")
    train['train_or_test']='train'
    test['train_or_test']='test'

    df=pd.concat([train,test])

    text.insert(END, "Combined dataset shape: {0}: ".format(df.shape)+"\n")

    df["maximum_hours"] = df["hour"].isin([8, 12,13,14,15,16,17,18]).astype("object")
    
    #postcode
    cat_features = ['postcode']

    # Create the encoder
    count_enc = ce.CountEncoder()

    # Transform the features, rename the columns with the _count suffix, and join to dataframe
    df['postcode_cnt'] = count_enc.fit_transform(df[cat_features])

    #Local_Authority_(Highway)
    cat_features_1 = ['Local_Authority_(Highway)']

    # Create the encoder
    count_enc = ce.CountEncoder()

    df['LAH_cnt'] = count_enc.fit_transform(df[cat_features_1])

    #Local_Authority_(District)
    cat_features_2 = ['Local_Authority_(District)']

    # Create the encoder
    count_enc = ce.CountEncoder()

    # Transform the features, rename the columns with the _count suffix, and join to dataframe
    df['LAD_cnt'] = count_enc.fit_transform(df[cat_features_2])

    #Police_Force
    cat_features_3 = ['Police_Force']

    # Create the encoder
    count_enc = ce.CountEncoder()

    # Transform the features, rename the columns with the _count suffix, and join to dataframe
    df['PF_cnt'] = count_enc.fit_transform(df[cat_features_3])

    df["maximum_hours"] = df["hour"].isin([8, 12,13,14,15,16,17,18]).astype("object")
    df['Season'] = df['month_in_year'].apply(month2seasons)

def dataEncoding():
    text.delete('1.0', END)
    global train, test, sample_submission , population, roads_network, df, train_corr
    encoding()
    train=df.loc[df.train_or_test.isin(['train'])]
    test=df.loc[df.train_or_test.isin(['test'])]
    train[object_columns] = train[object_columns].astype(object)
    test[object_columns] = test[object_columns].astype(object)
    
    text.insert(END, "data encoded successfully")

def coordinates():
    global train, test, sample_submission , population, roads_network, df, train_corr

    roads_network["WKT"] = roads_network["WKT"].str.replace('[a-zA-Z]', '')
    roads_network["WKT"]= roads_network["WKT"].replace(r'[(]+', ' ', regex=True)
    roads_network["WKT"]= roads_network["WKT"].replace(r'[)]+', ' ', regex=True)
    roads_network = roads_network.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # roads_network['Longitude'], roads_network['Latitude'] = roads_network['WKT'].str.split(' ', 1).str
    coordinates= roads_network['WKT'].str.split(' ')
    for i in range(len(coordinates)):
        roads_network['Longitude'], roads_network['Latitude'] = coordinates[i][2], coordinates[i][3]
    src_prj = _proj.Proj("+init=EPSG:4326")
    dst_prj = _proj.Proj("+init=EPSG:3857")

    x_coords = roads_network.Longitude.values
    y_coords = roads_network.Latitude.values
    x_coords, y_coords = _proj.transform(src_prj, dst_prj, x_coords, y_coords)
    roads_network["x"] = x_coords
    roads_network["y"] = y_coords
    roads_network.drop_duplicates(subset=['postcode'], keep='first', inplace=True)

    text.insert(END, "coordintates sets successfully...........")

def dataConcatenation():
    global x_train, X_test, y_train, y, x, train, test, train_corr
    text.delete('1.0', END)
    train = pd.merge(train, population, on = ['postcode'],how='left')
    test  = pd.merge(test, population, on = ['postcode'],how='left')

    train = pd.merge(train, roads_network, on = ['postcode'],how='left')
    test  = pd.merge(test, roads_network, on = ['postcode'],how='left')

    drop_columns_1 = ["Rural Urban","WKT","roadFuncti","formOfWay"]
    train.drop(drop_columns_1, axis=1, inplace=True)
    test.drop(drop_columns_1, axis=1, inplace=True)


    train.replace(to_replace=np.nan, value='999', inplace = True)
    test.replace(to_replace=np.nan, value='999', inplace = True)

    train[int_columns] = train[int_columns].astype(int)
    test[int_columns]  = test[int_columns].astype(int)

    train[float_columns] = train[float_columns].astype(float)
    test[float_columns]  = test[float_columns].astype(float)
    train['train_or_test']='train'
    test['train_or_test']='test'

    df=pd.concat([train,test])

    df["density_by_Police_Force"] = df["Variable: Density (number of persons per hectare); measures: Value"]/df["Police_Force"]
    df["population_by_Police_Force"] = df["Variable: All usual residents; measures: Value"]/df["Police_Force"]

    train=df.loc[df.train_or_test.isin(['train'])]
    test=df.loc[df.train_or_test.isin(['test'])]

    train['Date'] = train['Date'].astype('datetime64[ns]').astype(np.int64)/10**9
    test['Date'] = test['Date'].astype('datetime64[ns]').astype(np.int64)/10**9

    train['var_max_lat'] = train['Latitude'].max() - train['Latitude']
    train['var_max_long'] = train['Longitude'].max() - train['Longitude']
    test['var_max_lat'] = test['Latitude'].max() - test['Latitude']
    test['var_max_long'] = test['Longitude'].max() - test['Longitude']

    train['lon_plus_lat'] = train['Longitude'] + train['Latitude']
    test['lon_plus_lat'] = test['Longitude'] + test['Latitude']

    x_train=train.drop("Number_of_Casualties",axis=1)
    y_train=train["Number_of_Casualties"]

    train_corr = train.copy()
    train_corr.drop(["Accident_ID","postcode","year","country", "Time", "2nd_Road_Number","Day_of_Week","Local_Authority_(Highway)","Local_Authority_(District)","Police_Force","train_or_test"],axis=1,inplace=True)

    x_train.drop(["Accident_ID","postcode","year","country", "Time", "2nd_Road_Number","Day_of_Week","Local_Authority_(Highway)","Local_Authority_(District)","Police_Force","train_or_test"],axis=1,inplace=True)
    test.drop(["Accident_ID","postcode","year","country", "Time", "2nd_Road_Number","Day_of_Week","Local_Authority_(Highway)","Local_Authority_(District)","Police_Force","train_or_test"],axis=1,inplace=True)
    ordinal_encoding()

    corr_matrix=train_corr.corr()
    corr_matrix.sort_values('Number_of_Casualties')['Number_of_Casualties']

    y      = train["Number_of_Casualties"]
    x      = x_train.copy()
    X_test = test.copy()

    text.insert(END, "data successfully concatenated.........")

def ordinal_encoding():
    global x_train, test, train_corr
    cols=['Speed_limit','2nd_Road_Class','1st_Road_Class','Urban_or_Rural_Area','Did_Police_Officer_Attend_Scene_of_Accident']
    ord_en=OrdinalEncoder()
    x_train[cols]=ord_en.fit_transform(x_train[cols])
    test[cols]=ord_en.transform(test[cols])
    train_corr[cols]=ord_en.transform(train_corr[cols])

    #Creating dummy variables
    x_train    = pd.get_dummies(x_train, drop_first=True)
    test       = pd.get_dummies(test,drop_first=True)
    train_corr = pd.get_dummies(train_corr, drop_first=True)

def encoding():
    global df
    def frequency_encoding(column_name,output_column_name,df):
        fe_pol = (df.groupby(column_name).size()) / len(df)
        df[output_column_name] = df[column_name].apply(lambda x : fe_pol[x])
    comb = combinations(columns, 2)

    for i in list(comb):
        df[f'{i[0]}_{i[1]}']=df[i[0]].astype(str)+'_'+df[i[1]].astype(str)
        df[f'{i[0]}_{i[1]}_le']=le.fit_transform(df[f'{i[0]}_{i[1]}'])
        frequency_encoding(f'{i[0]}_{i[1]}',f'{i[0]}_{i[1]}',df)
        cat_features.append(f'{i[0]}_{i[1]}')

    frequency_encoding('1st_Road_Class','1st_Road_Class_fe',df)
    frequency_encoding('1st_Road_Number','1st_Road_Number_fe',df)
    frequency_encoding('2nd_Road_Class','2nd_Road_Class_fe',df)
    frequency_encoding('Speed_limit','Speed_limit_fe',df)
    frequency_encoding('Urban_or_Rural_Area','Urban_or_Rural_Area_fe',df)
    frequency_encoding('Number_of_Vehicles','Number_of_Vehicles_fe',df)
    frequency_encoding('Road_Type','Road_Type_fe',df)

def time_of_day(n):
    if n in range(4,8):
        return 'Early Morning'
    elif n in range(8,12):
        return 'Morning'
    elif n in range(12,17):
        return 'Afternoon'
    elif n in range(17,20):
        return 'Evening'
    elif n in range(20,25) or n==0:
        return 'Night'
    elif n in range(1,4):
        return 'Late Night'

def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    return season


def train_algorithm():
    global model
    text.delete("1.0",END)
    if os.path.exists('Model/model_weights.h5'):
        model.load_weights('Model/model_weights.h5')
        text.insert(END, "model loaded successfully......................")

    model_xgb = xgb.XGBRegressor(random_state =7)
    model_lgb = lgb.LGBMRegressor(random_state =7)
    model_cb  = CatBoostRegressor(random_state =7, verbose=False)
    model_rf  = RandomForestRegressor(random_state =7)
    score = rmsle_cv(model_cb)
    text.insert(END, "Catboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())+"\n")
    score = rmsle_cv(model_xgb)
    text.insert(END, "Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())+"\n")
    score = rmsle_cv(model_rf)
    text.insert(END, "RF score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())+"\n")
    debug=2000
    rounds=90000
    early_stop=20

    X_1 = x.values
    X_test_1 = X_test.values
    target = y.values.reshape(-1,1)
    verbose=debug

    train_oof = np.zeros((X.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))
    train_oof.shape,test_preds.shape

    #params = {'learning_rate': 0.017, 'max_depth': -1,'n_estimators': rounds,'metric': 'rmse',}
    params = {'bootstrap_type': 'Bernoulli','n_estimators': rounds,'eval_metric': 'RMSE','learning_rate': 0.017}
    #params = {'eval_metric': 'rmse',}
    n_splits = 10
    kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

    for jj, (train_index, val_index) in enumerate(kf.split(X_1, target)):
        text.insert(END, "Fitting fold", jj+1+"\n")
        train_features = X_1[train_index]
        train_target = target[train_index]

        val_features = X_1[val_index]
        val_target = target[val_index]

        #model = LGBMRegressor(**params)
        #model = XGBRegressor(**params)
        model = CatBoostRegressor(**params)
        hist = model.fit(train_features, train_target,
                eval_set=(val_features, val_target),
                early_stopping_rounds=early_stop,
                verbose=debug,
                )
        model.save_weights('Model/model_weights.h5')            
        model_json = model.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        val_pred = model.predict(val_features)
        train_oof[val_index] = val_pred.flatten()
        test_preds += model.predict(X_test_1).flatten()/n_splits

    text.insert(END,"mean_squared_error:",mean_squared_error(target,train_oof, squared=False)+"\n")
    text.insert(END,"r2_score:",r2_score(target,train_oof)+"\n")

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(y.values)
    r2= cross_val_score(model, x.values, y.values, scoring  = 'neg_root_mean_squared_error', cv = kf)
    return(r2)

def plot_feature_importance(importance,names,model_type):
    global model
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(15,55))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

    plot_feature_importance(model.get_feature_importance(),x_train.columns,'CATBOOST')

def predict():
    global test_preds
    text.delete('1.0', END)
    dataset = filedialog.askopenfile()
    df1=pd.read_csv(dataset)

    df1["casualities"]=test_preds

    Accident_risk_index=df1.groupby("postcode").casualities.sum()/df1.groupby("postcode").Accident_ID.count()

    postcode=list(df1.groupby("postcode").postcode.count().index)

    sub_final_df=pd.DataFrame({"postcode":postcode,"Accident_risk_index":Accident_risk_index})
    sub_final_df.to_csv("cb_new_divide_by_ft_pluslat_lon_reduced.csv",index=False)
    with open("cb_new_divide_by_ft_pluslat_lon_reduced.csv", "r") as f:
        data = f.read()
        text.insert(END, data)

def on_change(*args):
    text.delete('1.0',END)
    typed_text = selected_value.get().upper()
    matched_options = [option for option in options if typed_text in option.upper()]
    dropdown['values'] = matched_options
    result = dataset[dataset['postcode'] == typed_text]['Accident_risk_index']
    if len(result)!= 0:
        text.insert(END, "postCode  :  {0} \t\t\t Accident_risk_index  :  {1} ".format(typed_text, result.values[0])+"\n")

font = ('times', 20, 'bold')
title = Label(main, text='Accident risk score prediction using ML')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=2, width=80)       
title.place(x=0,y=1)

font1 = ('times', 16, 'bold')
text=Text(main,height=17,width=89)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)

# text_widget= Text(main, height=1, width=50, font=("Arial", 18))
# text_widget.place(x=50,y=120)

dataset = pd.read_csv("dataset/sample_submission.csv")
options = dataset['postcode'].unique()

selected_value = tk.StringVar()
dropdown = ttk.Combobox(main, textvariable=selected_value, values=options, font=("Arial", 14),height=15)
dropdown.set("Select a postcode")
dropdown.pack(pady=20)
dropdown.place(x=1040, y=120)
dropdown.config()

selected_value.trace_add("write", on_change)

def on_click(event):
    dropdown.set('')
    dropdown.unbind("<Button-1>", on_click)
dropdown.bind("<Button-1>", on_click)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset cleaning", command= datasetCleaning)
preprocessButton.place(x=250,y=550)
preprocessButton.config(font=font1)

dataencoding = Button(main, text="Dataset encoding", command= dataEncoding)
dataencoding.place(x=450,y=550)
dataencoding.config(font=font1)

coordinate = Button(main, text="set coordinates", command= coordinates)
coordinate.place(x=650,y=550)
coordinate.config(font=font1)

dataconcatenation = Button(main, text="Dataset concatenation", command= dataConcatenation)
dataconcatenation.place(x=850,y=550)
dataconcatenation.config(font=font1)

cnnButton = Button(main, text="Training Model", command=train_algorithm)
cnnButton.place(x=1050,y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="predicting accident through dataset", command= predict)
classifyButton.place(x=300,y=610)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=plot_feature_importance)
graphButton.place(x=700,y=610)
graphButton.config(font=font1)

main.config(bg='light pink')
main.mainloop()