from datetime import timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.stattools import acf,pacf, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import time
import logging
import pickle

class PredictionModel:

    __model=None
    __data=None
    __scaler=None
    __exog_columns_subset=[]
    __last_index=None
    __logger = logging.getLogger('uvicorn.error')

    def read_verify(self) -> DataFrame | None:
        PredictionModel.__logger.info("Reading data")
        data = pd.read_csv("SeoulBikeData.csv")
        # with open("debug.txt","w") as file:
        #     file.write(data.head(5).to_csv())
        missing_data_count=data.isna().sum().sum()
        if(missing_data_count>0):
            raise Exception("There is missing data")
        return data

    def preprocess_data(self,data: DataFrame) -> DataFrame | None:
        PredictionModel.__logger.info("Preprocessing data")
        data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
        data["Date"] = data["Date"] + pd.to_timedelta(data["Hour"], unit="h")
        data = data.set_index(data["Date"])
        categorical_cols = ["Seasons", "Holiday", "Functioning Day"]
        data[categorical_cols] = data[categorical_cols].astype("category")

        data = pd.get_dummies(data, columns=["Seasons"], dtype=int)
        holiday_map = {"Holiday": 0, "No Holiday": 1}
        functioning_day_map = {"Yes": 1, "No": 0}
        data["Holiday"] = data["Holiday"].map(holiday_map)
        data["Functioning Day"] = data["Functioning Day"].map(functioning_day_map)

        columns_renamed = {"Rented Bike Count": "target", "Temperature(C)": "temp", "Humidity(%)": "hum",
                           "Wind speed (m/s)": "wind", "Visibility (10m)": "vis",
                           "Dew point temperature(C)": "dew", "Solar Radiation (MJ/m2)": "solar", "Rainfall(mm)": "rain",
                           "Snowfall (cm)": "snow", "Holiday": "holiday",
                           "Functioning Day": "func", "Seasons_Autumn": "s_autumn", "Seasons_Spring": "s_spring",
                           "Seasons_Summer": "s_summer", "Seasons_Winter": "s_winter"}
        data = data.rename(columns=columns_renamed)
        data = data[[columns_renamed[old_column_name] for old_column_name in columns_renamed.keys()]]
        return data

    def adf_test(self,data: DataFrame) -> None:
        PredictionModel.__logger.info("ADF Test")
        data["first_diff"] = data["target"].diff()
        adf_test = adfuller(data["first_diff"][1:])
        p_value=adf_test[1]
        if(p_value>0.05):
            raise Exception("First difference is not stationary")

    def scale_columns(self,data: DataFrame):
        PredictionModel.__logger.info("Scaling columns")
        scaling_columns = self.get_exog_columns()
        scaler = MinMaxScaler()
        data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
        with open('scaler.pkl','wb') as file:
            pickle.dump(scaler,file)
        return data

    def get_arima_orders(self):
        PredictionModel.__logger.info("Get ARIMA orders")
        non_seasonal_orders = (1, 1, 0)
        seasonal_orders = (1, 0, 0, 24)
        return non_seasonal_orders,seasonal_orders

    def get_exog_columns(self):
        PredictionModel.__logger.info("Get Exogenous columns")
        exog_columns_subset = ['temp', 'hum', 'vis', 'func']
        with open('exog_cols.pkl','wb') as file:
            pickle.dump(exog_columns_subset,file)
        return exog_columns_subset

    def train_arima(self,data: DataFrame,non_seasonal_orders,seasonal_orders,exog_columns):
        PredictionModel.__logger.info("Training ARIMA")
        ts_model_sarimax = SARIMAX(data["target"], order=non_seasonal_orders,
                                                     seasonal_order=seasonal_orders,
                                                     exog=data[exog_columns])

        start = time.time()
        model_fit= ts_model_sarimax.fit()
        end = time.time()
        PredictionModel.__logger.info("Model Fitting Time: "+ str(end - start))
        return model_fit

    async def forecast(self,number_of_periods,input_data):
        PredictionModel.__logger.info("Forecasting with parameters: "+str({'periods':number_of_periods,'input':input_data}))
        with open('last_index.pkl','rb') as file:
            last_index=pickle.load(file)
        with open('exog_cols.pkl','rb') as file:
            exog_columns=pickle.load(file)
        with open('scaler.pkl','rb') as file:
            scaler=pickle.load(file)
        with open('bike_rental_demand_prediction_model.pkl','rb') as file:
            model=pickle.load(file)
        future_indices=pd.date_range(start=last_index+timedelta(hours=1),periods=number_of_periods,freq='h')

        input_data_df=pd.DataFrame(data=input_data,index=future_indices)
        input_data_df=pd.DataFrame(data=scaler.transform(input_data_df[exog_columns]),columns=exog_columns,index=future_indices)
        prediction_detail=model.get_forecast(steps=number_of_periods,exog=input_data_df.loc[:,exog_columns])
        predicted_mean=prediction_detail.predicted_mean
        predicted_confidence_intervals=prediction_detail.conf_int()
        indices=predicted_mean.index
        predicted_mean=predicted_mean.tolist()

        lower_limits=predicted_confidence_intervals['lower target'].tolist()
        upper_limits=predicted_confidence_intervals['upper target'].tolist()
        data=[[predicted_mean[i],lower_limits[i],upper_limits[i]] for i in range(len(predicted_mean))]
        response=pd.DataFrame(data=data,columns=['Value','Lower Limit','Upper Limit'],index=indices)
        return response

    async def train_model(self):
        PredictionModel.__model=self.pipeline()

    def pipeline(self):
        PredictionModel.__logger.info("Training Pipeline Started")
        data=self.read_verify()
        data=self.preprocess_data(data)
        self.adf_test(data)
        data=self.scale_columns(data)
        with open('data.pkl','wb') as file:
            pickle.dump(data,file)
        with open('last_index.pkl','wb') as file:
            pickle.dump(data.index[-1],file)
        non_seasonal_orders,seasonal_orders=self.get_arima_orders()
        exog_columns=self.get_exog_columns()
        model=self.train_arima(data,non_seasonal_orders,seasonal_orders,exog_columns)
        model.save("bike_rental_demand_prediction_model.pkl")
        return model


