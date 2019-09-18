#======= Import the neccessary modules ========================================
import os
#import re
import wx             # Install it with pip.
import sys
import json
#import shutil         # Install it with pip.
import numpy as np
import pandas as pd
#from os import path
#from datetime import datetime
#from sklearn.utils import resample
# Instantiate and Consume the class.
from classMLModelingPipeline import *

#import matplotlib.pyplot as plt

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')

# Used for logging.
s_classApplication  = 'classMLModelingPipeline'
# Set the parameters to determine the model hyper parameters.
classWt = 1      # 0=>None|1=>Balanced
penalty = 'l2'   # optimizer
cv      = 5      # Sampling epochs.
solver  = 'liblinear' # Solver.

#==============================================================================
#======== Function to prompt the user to upload the file. =====================
def get_path(wildcard):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Choose a training file.', 
                           wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path
#==============================================================================
def instantiateMLPipleline():
    try:
        # Set the log variables.
        s_classMethod       = 'CreateAppDirectories'
        s_statusType        = 'Success'
        s_statusDescription = 'Successfull write operation.'
        # Call the funtion to get the input files.
        #s_infile = get_path('*.csv')
        global s_infile
        s_infile = get_path('*.xlsx')
        s_infile = s_infile.replace("\\","\\\\")
    
        # Get the current working directory.
        s_path = os.getcwd()
        s_path = s_path.replace("\\","\\\\")
        s_basefile = os.path.basename(s_infile)
        global s_basefile_csv
        s_basefile_csv = (os.path.splitext(s_basefile)[0]) + '.csv'
    
        # Check, if path exists in system path,else add.
        if s_path in os.environ:
            sys.path.append(s_path)
    
        # Open and read the App Configuration using json.
        with open(s_path + '\\AppConfig.txt') as json_file:
            # Load the App config details.
            data = json.load(json_file)
            # For each entry in json, extract App config parameters.
            for p in data['AppConfig']:
                global applConfig
                applConfig = modMLModelingPipeline  (  p['Id'],
                                                        p['Name'],
                                                        p['Source'],
                                                        p['Output'],
                                                        p['TrainedModel'],
                                                        p['ExecutionLog'],
                                                        p['ExecutionLogFileName'],
                                                        p['FeedActiveSheet'],
                                                        p['FeedSkipRows'],
                                                        p['Archive']
                                                      )
        #===== Create Application Directories ============================================================
        applConfig.CreateAppDirectories()
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )
    except AttributeError:
        s_statusDescription = 'No file has been choosen!!!!' 
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
    except:
        s_statusDescription = 'Unexpected error : ' 
        s_statusDescription = s_statusDescription + str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
def read_fromcsv():
    try:
        s_classMethod       = 'Data Loading'    
        s_statusType        = 'Success'
        feed_active_sheet   = applConfig.FeedActiveSheet
        global feed_skiprows
        feed_skiprows       = int(applConfig.FeedSkipRows)
        feed_skiprows_list  = list(range(int(feed_skiprows)))
        if len(feed_skiprows_list) > 0:
            # Read the input file.
            data_xls = pd.read_excel(s_infile, feed_active_sheet, index_col=None,skiprows=feed_skiprows_list)
        else:
            data_xls = pd.read_excel(s_infile, feed_active_sheet, index_col=None)
        # Drop empty rows.
        data_xls.dropna(axis=0,how='any',inplace=True)
        # Get the path to source directory from class.
        s_in_source_file = applConfig.Source
        s_in_source_file = s_in_source_file + s_basefile_csv
    
        # Convert it to .csv utf-8 format.
        data_xls.to_csv(s_in_source_file, encoding='utf-8')
        
        # Load the data from the csv file into pandas dataframe.
        dfDataVendor = pd.read_csv(s_in_source_file
                                   ,parse_dates = ['Doc. Date','Del Date','GRN Date']
                                   ,infer_datetime_format = True
                                  )
        # Drop the unwanted column.
        dfDataVendor.drop(columns=['Unnamed: 0'],inplace=True)
        # Extract VendorCode from "Supplier/Supplying Plant" column.
        dfDataVendor['VendorCode']  = dfDataVendor['Supplier/Supplying Plant'].str.extract('(\d+)')
        # Add code to split VendorCode and Vendor Name.
        # Add code to split MaterialCode and Material Name.
        
        return dfDataVendor
    except:
        s_statusType        = 'Error'    
        s_statusDescription = 'Unexpected error : ' 
        s_statusDescription = s_statusDescription + str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)    
#===================================================================================================
def GetValidExceptionForDates(lstDateColumns,  # list of date columns to validate.
                              dfData,          # data frame of the input data.
                              infile_skiprows  # number of rows to skip in the begining from the input file.
                             ):
    # Initialize.
    validationException = ''
    validException_item = ''
    # Corrected Row_index in source file.
    cRowIndex = infile_skiprows + 2
    # Build the Validation Exception message.
    validExceptionPresent = 0
    #validExcept_header = "'ValidationException':"
    validExcept_body   = "'Incorrect_DateFormat' : ["
    validExcept_bodyfooter = "]"
    # Loop through the list of Date columns.
    for itm in lstDateColumns:
        # Validate columns.
        dfDateCheck = pd.to_datetime(dfData[itm],errors='coerce')
        # Get the row indices, where the validation exception occurred.
        lstDateExceptions = list(dfDateCheck[dfDateCheck.isnull()].index+cRowIndex)
        if len(lstDateExceptions) > 0:
            validExceptionPresent = 1
            validException_item = validException_item + "'" + itm + "':" + str(lstDateExceptions) + ","

    if validExceptionPresent:
        # Remove extra comma.
        validException_item = validException_item[:-1]
        # Build the Validation Exception Message.
        validationException = validExcept_body + validException_item + validExcept_bodyfooter
    return validationException
#===================================================================================================
def GetValidExceptionForNumeric(lstNumericColumns,  # list of date columns to validate.
                                dfData,          # data frame of the input data.
                                infile_skiprows  # number of rows to skip in the begining from the input file.
                               ):
    # Initialize.
    validationException = ''
    validException_item = ''
    # Corrected Row_index in source file.
    cRowIndex = infile_skiprows + 2
    # Build the Validation Exception message.
    validExceptionPresent = 0
    #validExcept_header = "'ValidationException':"
    validExcept_body   = "'Incorrect_NumericFormat' : ["
    validExcept_bodyfooter = "]"
    # Loop through the list of Date columns.
    for itm in lstNumericColumns:
        # Validate columns.
        dfNumericCheck = pd.to_numeric(dfData[itm],errors='coerce')
        lstNumericExceptions = list(dfNumericCheck[dfNumericCheck.isnull()].index+cRowIndex)
        if len(lstNumericExceptions) > 0:
            validExceptionPresent = 1
            validException_item = validException_item + "'" + itm + "':" + str(lstNumericExceptions) + ","

    if validExceptionPresent:
        # Remove extra comma.
        validException_item = validException_item[:-1]
        # Build the Validation Exception Message.
        validationException = validExcept_body + validException_item + validExcept_bodyfooter
    return validationException
#===================================================================================================
def GetAllExceptions(dfDataVendor):
    try:
        # Consolidate error description to write to log.
        #======= Validate Numeric columns ========================================================================================
        raise_Except = 0
        validationException = ''
        strComma = ''
        validExcept_header = "'ValidationException':"
        s_statusType        = 'ValidationException'    
        s_classMethod       = 'Parse Datatypes'
        # Convert date column's datatype from string to datetime.
        lstNumericCols = ['Sum of PO Quantity','Sum of      Net Price','Sum of PO Value', 'Sum of GRN Qty','Sum of GRN Val']
        exceptMessage = GetValidExceptionForNumeric(lstNumericColumns = lstNumericCols, # list of numeric columns to validate.
                                                  dfData = dfDataVendor,             # data frame of the input data.
                                                  infile_skiprows = feed_skiprows    # number of rows to skip in the begining from the input file.
                                                 )
        if len(exceptMessage) > 0:
            raise_Except = 1
            s_statusDescription = exceptMessage
    
        #======= Validate Date columns =====================================================================================
        # set the class name and status.
        # Convert date column's datatype from string to datetime.
        lstDateCols = ['Doc. Date','Del Date','GRN Date']
        # Get the validation exceptions.
        exceptMessage = GetValidExceptionForDates(lstDateColumns = lstDateCols, # list of date columns to validate.
                                                  dfData = dfDataVendor,           # data frame of the input data.
                                                  infile_skiprows = feed_skiprows  # number of rows to skip in the begining from the input file.
                                                 )
        # If validation exception found.
        if len(exceptMessage) > 0:
            if raise_Except == 1:
                strComma = ','
            raise_Except = 1
            s_statusDescription = s_statusDescription + strComma + exceptMessage
        
        if raise_Except:
            validationException = validExcept_header + '{' + s_statusDescription + '}'
            # Write to log.
            applConfig.WriteToActivityLog(s_classApplication,
                                          s_classMethod,
                                          s_statusType,
                                          s_statusDescription
                                         ) 
            
            raise Exception('Data validation error/s occured. Please check the application log.')
        
    except:
        s_statusType        = 'Error'
        s_statusDescription = validationException + str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)    
#===================================================================================================
def CalculateDeliveryPerformanceMetrics(dfDataVendorMaxDate):
    try:
        s_classMethod       = 'Calculate deviation metrics'
        s_statusType        = 'Success' 

        # Deviation in expected vs actual delivery dates.
        dfDataVendorMaxDate['Deviation_DeliveryDate'] = (dfDataVendorMaxDate['GRN Date'] - dfDataVendorMaxDate['Del Date']).dt.days
    
        # Deviation in expected vs actual delivered quantity.
        dfDataVendorMaxDate['Deviation_DeliveredQty'] = (dfDataVendorMaxDate['Sum of PO Quantity'] - dfDataVendorMaxDate['Sum of GRN Qty']) \
        /dfDataVendorMaxDate['Sum of PO Quantity']
    
        # Round of to the nearest 3rd decimal.
        #dfDataVendorMaxDate['Deviation_DeliveredQty'] = dfDataVendorMaxDate['Deviation_DeliveredQty'].round(3)
        dfDataVendorMaxDate['Deviation_DeliveredQty'] = np.abs(dfDataVendorMaxDate['Deviation_DeliveredQty'].round(3))
    
        # Deviation in expected vs actual delivered value.
        dfDataVendorMaxDate['Deviation_DeliveredValue'] = (dfDataVendorMaxDate['Sum of PO Value'] - dfDataVendorMaxDate['Sum of GRN Val']) \
        /dfDataVendorMaxDate['Sum of PO Value']
    
        # Round of to the nearest 3rd decimal.
        #dfDataVendorMaxDate['Deviation_DeliveredValue'] = dfDataVendorMaxDate['Deviation_DeliveredValue'].round(3)
        dfDataVendorMaxDate['Deviation_DeliveredValue'] = np.abs(dfDataVendorMaxDate['Deviation_DeliveredValue'].round(3))
    
        # Handle negative deviations in quantity and value. Delivered more qty/value than the PO contract.
        dfDataVendorMaxDate.loc[dfDataVendorMaxDate['Deviation_DeliveredQty']   < 0.0,  'Deviation_DeliveredQty'] = 0.0
        dfDataVendorMaxDate.loc[dfDataVendorMaxDate['Deviation_DeliveredValue'] < 0.0,'Deviation_DeliveredValue'] = 0.0
        
        # Get the unique Material/Vendor combo.
        dfMaterialCountByVendor = dfDataVendorMaxDate[['Material','VendorCode']].drop_duplicates()
        # Get distinct material count supplied by Vendor.
        dfMaterialCountByVendor = dfDataVendorMaxDate.groupby(['VendorCode']).agg({'Material':'count'}) \
                                                     .reindex(['Material'],axis=1) \
                                                     .reset_index()
        # Rename the material column to MaterialCountByVendor in df dfMaterialCountByVendor
        dfMaterialCountByVendor.rename(columns={'Material':'MaterialCountByVendor'}, inplace=True)
        # Update the material count by vendor back to the main dataframe.
        dfDataVendorMaxDate = dfDataVendorMaxDate.merge(dfMaterialCountByVendor,on=['VendorCode'],how='inner')    

        # Concat all the columns required for the analysis.
        dfDataVendorMaxDate['Deviation_DeliveryDate'] = dfDataVendorMaxDate['Deviation_DeliveryDate'].round(0)
        dfDataVendorMaxDate['Deviation_DeliveredQty'] = dfDataVendorMaxDate['Deviation_DeliveredQty'].round(2)
        dfDataVendorFinal  = dfDataVendorMaxDate[['Doc. Date','VendorCode','Material','Sum of      Net Price', \
                                                'Sum of GRN Qty','Sum of GRN Val','Sum of PO Quantity','Sum of PO Value', \
                                                'Deviation_DeliveryDate','Deviation_DeliveredQty', \
                                                'Deviation_DeliveredValue','MaterialCountByVendor']].copy()
        
        # Dataframe to hold the delivery performance parameters to be pickled.
        dfPickleDeliveryPerformance = dfDataVendorFinal[['Material','VendorCode','Deviation_DeliveryDate',\
                                                         'Deviation_DeliveredQty','Deviation_DeliveredValue', \
                                                         'MaterialCountByVendor']].copy()
        # Return the dataframes.
        return dfDataVendorFinal,dfPickleDeliveryPerformance
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
# Function to generate the class Label.
def getClassDelivery(dfClassDelivery):
    # Store the actual index.
    dfIndex = dfClassDelivery.index
    # Set the index to 'Material' to partition the dataset.
    dfClassDelivery.set_index('Material', inplace=True)
    # Loop through the distinct partition value.
    for i in (set(dfClassDelivery.index)):
        # Apply partition, on the 'Material' column.
        slicer = (dfClassDelivery.index.values == i)
        # No deviation in delivery dates and quantity delivered, set the class 
        # label to 1.
        dfClassDelivery.loc[( (slicer) & \
                              (dfClassDelivery['Deviation_DeliveryDate']   <= 0.0) & \
                              (dfClassDelivery['Deviation_DeliveredQty']   == 0.0) \
                             
                            ),  \
                            'classDelivery'] = 1

    dfClassDelivery['Material'] = dfClassDelivery.index
    dfClassDelivery.index = dfIndex
    dfClassDelivery = dfClassDelivery[['Doc. Date','Material','VendorCode','Sum of      Net Price', \
                                       'Sum of GRN Qty','Sum of GRN Val','Sum of PO Quantity','Sum of PO Value', \
                                       'Deviation_DeliveryDate','Deviation_DeliveredQty','MaterialCountByVendor', \
                                       'Rank','classDelivery']]
    return dfClassDelivery
#===================================================================================================
# Function to apply class labels.
def generateClassLabels(dfDataVendorFinal):
    try:
        s_classMethod       = 'Generate Rank and Class Labels'
        s_statusType        = 'Success'        
        # prepare the rank data by hard coding the weights. 
        dfDataVendorFinal['Rank'] = ( (dfDataVendorFinal['Deviation_DeliveryDate']*0.4) + \
                                      (dfDataVendorFinal['Deviation_DeliveredQty']*0.4) + \
                                      (dfDataVendorFinal['MaterialCountByVendor']*0.2) 
                                    )
    
        # Call the function to generate the class label.
        # Set the flag to 'Has Deviation'
        # class_0_Has_Deviation | class_1_Has_No_Deviation
        dfDataVendorFinal['classDelivery'] = 0
        dfDataVendorFinal = getClassDelivery(dfDataVendorFinal.copy())
        # Return the dataframe.
        return dfDataVendorFinal
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)         
#===================================================================================================
# Function to treat categorical variables.
def featureEngineerCategories(dfDataVendorFinal):
    try:
        s_classMethod       = 'Treat Categorical Varaibles'
        s_statusType        = 'Success' 
        # Build the dataframe with the encoded column for the categorical data.
        dfMaterialOHE = pd.get_dummies(dfDataVendorFinal['Material'])
        dfVendorOHE   = pd.get_dummies(dfDataVendorFinal['VendorCode'])
        # Add the encoded columns to the existing dataframe.
        dfDataVendorFinal = pd.concat([dfDataVendorFinal,dfMaterialOHE,dfVendorOHE],axis=1)
        return dfDataVendorFinal
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
# Function to split the dataset.
def split_data_train_test(dfDataVendorFinal):
    try:
        s_classMethod       = 'Perform Data Split'
        s_statusType        = 'Success' 
    
        # Copy the dataframe.
        dfDataVendorML = dfDataVendorFinal.copy()
    
        # Extract the dependent variable.
        Y_Output = dfDataVendorML[['classDelivery']].copy()
        # Remove features not useful for the modelling.
        featureDrop = ['classDelivery', \
                       'Doc. Date', \
                       'Material', \
                       'VendorCode', \
                       'Rank' \
                       ,'Sum of      Net Price' \
                       ,'Sum of PO Quantity' \
                       ,'Sum of PO Value' \
                       ,'Sum of GRN Qty' \
                       ,'Sum of GRN Val'
                       ,'MaterialCountByVendor'
                       ]
        dfDataVendorML.drop(featureDrop, axis=1,inplace=True)
        # Extract the independent variable.
        X_Input = dfDataVendorML.copy()
    
        # Get the total row count.
        total_rowCount = len(X_Input)
        # Is time based splitting of dataset required.
        IsTimeBasedSplitting = 0
        # Data split ratio for train, cv and test set.
        Training_split_ratio = 0.60
        Crossvalidation_split_ratio = 0.20
        Test_split_ratio = 0.20
    
        # Call the method to split the data.
        X_train,X_cv,X_test,Y_train,Y_cv,Y_test = applConfig.split_data(dfX = X_Input,
                                                                        dfY = Y_Output,
                                                                        total_rowCount = total_rowCount,
                                                                        IsTimeBasedSplitting = IsTimeBasedSplitting,
                                                                        Training_split_ratio = Training_split_ratio,
                                                                        Crossvalidation_split_ratio = Crossvalidation_split_ratio,
                                                                        Test_split_ratio = Test_split_ratio)
        # Return the splitted data.
        return X_train,X_cv,X_test,Y_train,Y_cv,Y_test
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
# Function to Standardize the data.
def standardize_data_train_test(X_train,X_cv,X_test,
                                Y_train,Y_cv,Y_test):
    try:
        s_classMethod       = 'Perform Data Split'
        s_statusType        = 'Success' 
        IsNormalize         = 1
        
        # Standardize the data.
        normalizer,X_train_stdzd,X_cv_stdzd, X_test_stdzd, \
        Y_train_ravel,Y_cv_ravel,Y_test_ravel = applConfig.standardize_data( X_train,
                                                                             X_cv,
                                                                             X_test,
                                                                             Y_train,
                                                                             Y_cv,
                                                                             Y_test,
                                                                             IsNormalize=IsNormalize)
        # Return the standardized data
        return normalizer,X_train_stdzd,X_cv_stdzd, X_test_stdzd, \
               Y_train_ravel,Y_cv_ravel,Y_test_ravel
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)             
#===================================================================================================
# Function to get model hyper-parameters.
def GetBestModelHyperparameters(X_train_stdzd,Y_train_ravel):
    try:
        s_classMethod       = 'Perform Hyper-parameter tuning'
        s_statusType        = 'Success' 
    
        # Get the model hyper-parameters.
        gridResults,bestScore, \
        optimal_HyperParameter = applConfig.GetModelHyperParameters(X_train_stdzd
                                                                    ,Y_train_ravel
                                                                    ,penalty  # Optimizer.
                                                                    ,cv       # GridSearchCV cv sample size
                                                                    ,classWt  # class-weight: 0=>None|1=>'balanced'
                                                                    ,solver # solver.
                                                                   )
        # Return the output.
        return gridResults,bestScore,optimal_HyperParameter
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
# Function to train the model.
def TrainModelOnData(X_train_stdzd, Y_train_ravel,
                     X_test_stdzd,  Y_test,
                     optimal_HyperParameter):
    try:
        s_classMethod       = 'Perform Model training'
        s_statusType        = 'Success' 
    
        # Get the trained model with the optimal hyperparameter.
        lr_optimal = applConfig.GetTrainedModel(X_train_stdzd,
                                                Y_train_ravel,
                                                optimal_HyperParameter,
                                                penalty,     # Optimizer
                                                solver,      # Solver
                                                classWt      # classWeight
                                               )
    
        # Get the calibrated model.
        calibratedCCV = applConfig.GetCalibratedModel( lr_optimal,
                                                       X_train_stdzd,
                                                       Y_train_ravel)
    
        # Get the predictions from the test set.
        Y_pred_test = applConfig.GetPredictions( lr_optimal,
                                                 X_test_stdzd)
    
        # Get the calibrated predictions from the test set.
        Y_pred_calib = applConfig.GetCalibratedPredictions( calibratedCCV,
                                                            X_test_stdzd)
    
        # Get the model confusion matrix
        #tickLabels = ['Deviation','No Deviation']
        plt , confmat, \
        tn, fp, fn, tp = applConfig.GetModelConfusionMatrixForBinaryClass(Y_test,
                                                                          Y_pred_test)
        # Return the model metrics.
        return lr_optimal,calibratedCCV,Y_pred_calib,confmat,plt,tn, fp, fn, tp
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)      
#===================================================================================================
# Function to get ROC score.
def GetModelROC(Y_test,Y_test_ravel,Y_pred_calib):
    try:
        s_classMethod       = 'Perform Model evaluation'
        s_statusType        = 'Success' 

        # Get the actual class distribution from the test set.
        negative_class = Y_test['classDelivery'].value_counts()[0]
        positive_class = Y_test['classDelivery'].value_counts()[1]        
        # Get the ROC value.
        plt,roc_score = applConfig.GetModelROCForBinaryClass(Y_test_ravel,Y_pred_calib)

        # Return ROC score.
        return roc_score,negative_class,positive_class
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
# Function to log model metrics.
def LogModelMetrics(gridResults,optimal_HyperParameter,confmat,
                    roc_score,positive_class,negative_class):
    try:
        s_classMethod       = 'Log Model Metrics'
        s_statusType        = 'Success'
        
        # Generate Model trace.
        # Set the model trace parameters.
        modelTrace = "[ModelHyperParameters]:[" + str(gridResults) + "];"
        modelTrace = modelTrace + "[ModelOptimalHyperParameter]:[" + str(optimal_HyperParameter) + "];"
        modelTrace = modelTrace + "[Class_Labels]:[0|1];"
        modelTrace = modelTrace + "[ModelConfusionMatrix]:[" + str(confmat) + "];"
        modelTrace = modelTrace + "[ModelROC_AUC]:[" + str(roc_score) + "];"
        modelTrace = modelTrace + "[Actual_ClassDistribution]:[" + str(positive_class) + "|" + str(negative_class) + "];"
    
        # Call method to write the model trace to log.
        applConfig.WriteToActivityLog(classApplication  = s_classApplication,
                                      classMethod       = 'Log Model Metrics',
                                      statusType        = 'Success',
                                      statusDescription = modelTrace)
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)  
#===================================================================================================
# Function to pickle the trained model.
def PickleTrainedObject(normalizer,lr_optimal,
                        calibratedCCV,dfPickleDeliveryPerformance):
    try:
        s_classMethod       = 'Pickle Trained Model'
        s_statusType        = 'Success'
        # Pickle the standardizer.
        pklTrainedStandardizer = applConfig.PickleTrainedObject('TN',normalizer)
        # Pickle the trained Model.
        pklTrainedModel = applConfig.PickleTrainedObject('TM',lr_optimal)
        #print(pklTrainedModel)
        pklTrainedCalibratedModel = applConfig.PickleTrainedObject('TCM',calibratedCCV)
        #print(pklTrainedCalibratedModel)
        pklDeliveryPerformanceParameters = applConfig.PickleDataframes(dfPickleDeliveryPerformance)
        #print(pklDeliveryPerformanceParameters)
        
        # Return the object.
        return pklTrainedStandardizer, \
               pklTrainedModel,pklTrainedCalibratedModel, \
               pklDeliveryPerformanceParameters
    except:
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)  
#===================================================================================================
# Call the functions when this python file is invoked.
if __name__ == "__main__":
    # Instantiate the ML class.
    instantiateMLPipleline()
    # Read data from csv.
    dfCV = read_fromcsv()
    # Check for exceptions in data.
    GetAllExceptions(dfCV)
    # Calculate delivery performance parameters.
    dfDV,dfDPP = CalculateDeliveryPerformanceMetrics(dfCV)
    # Generate class labels.
    dfDVClass = generateClassLabels(dfDV)
    # One Hot encode categories.
    dfDVClassCat = featureEngineerCategories(dfDVClass)
    # Split data into train,cv and test.
    X_train,X_cv,X_test,Y_train,Y_cv,Y_test = split_data_train_test(dfDVClassCat)
    # Standardize/Normalize the data.
    normalizer,X_train_stdzd,X_cv_stdzd, X_test_stdzd,Y_train_ravel,Y_cv_ravel,Y_test_ravel = \
    standardize_data_train_test(X_train,X_cv,X_test,Y_train,Y_cv,Y_test)
    # Hyper-parameter tuning of model parameters.
    gridResults,bestScore,optimal_HyperParameter = \
    GetBestModelHyperparameters(X_train_stdzd,Y_train_ravel)
    # Train the model.
    lr_optimal,calibratedCCV,Y_pred_calib,confmat,plt,tn, fp, fn, tp = \
    TrainModelOnData(X_train_stdzd, Y_train_ravel,X_test_stdzd,  Y_test,optimal_HyperParameter)
    # Get Model ROC Score.
    roc_score,negative_class,positive_class = GetModelROC(Y_test,Y_test_ravel,Y_pred_calib)
    # Log model metrics.
    LogModelMetrics(gridResults,optimal_HyperParameter,confmat, \
                    roc_score,positive_class,negative_class)
    # Pickle the trained objects.
    pklTrainedStandardizer,pklTrainedModel,pklTrainedCalibratedModel, \
    pklDeliveryPerformanceParameters = PickleTrainedObject(normalizer,lr_optimal,
                                                           calibratedCCV,dfDPP)