# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:33:30 2019
@author: Joseph MT Vijay
"""
#======= Import the neccessary modules ========================================
# Import the required modules.
import os
import re
import wx             # Install it with pip.
import sys
import json
import shutil         # Install it with pip.
import numpy as np
import pandas as pd
from os import path
from datetime import datetime
from sklearn.utils import resample
# Instantiate and Consume the class.
from classMLModelingPipeline import *

import matplotlib.pyplot as plt
#%matplotlib inline

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')
#==============================================================================
# Used for logging.
s_classApplication  = 'classMLModelingPipeline-Prediction'

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
        s_statusDescription = 'Successfully ranked vendords.'
        # Call the funtion to get the input files.
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
        raise Exception(s_statusDescription)
    except:
        s_statusDescription = 'Unexpected error : ' 
        s_statusDescription = s_statusDescription + str(sys.exc_info())
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
                                   ,parse_dates = ['Doc. Date','Del Date']
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
        lstNumericCols = ['Sum of PO Quantity','Sum of      Net Price','Sum of PO Value']
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
        lstDateCols = ['Doc. Date','Del Date']
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
def RollupMultipleMaterialVendorCombo(dfDataVendor):
    try:
        s_classMethod       = 'Resolve Multiple Material-Vendor combo'
        s_statusType        = 'Success'    
        # Get the latest data for material/combo appearing more than once.
        dfDataVendorMaxDate = dfDataVendor.groupby(['Material', \
                                                    'UOM', \
                                                    'VendorCode']).agg({'Doc. Date':'max', \
                                                                        'Del Date' : 'max', \
                                                                        'Sum of PO Quantity':'mean', \
                                                                        'Sum of      Net Price':'mean', \
                                                                        'Sum of PO Value' : 'mean' \
                                                                        }) \
                              .reindex(['Doc. Date','Del Date', \
                                        'Sum of PO Quantity', \
                                        'Sum of      Net Price', \
                                        'Sum of PO Value' \
                                       ], axis=1) \
                              .reset_index()
        return dfDataVendorMaxDate
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
def GetDeliveryPerformanceParameters():
    try:
        s_classMethod       = 'Get the deviation metrics'
        s_statusType        = 'Success'    
    
        # Get the delivery performance parameters from the pickle folder.
        dfDeliveryPerformanceParameters = applConfig.ReadDataframeFromPickle()
        return dfDeliveryPerformanceParameters
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
def GetUniqueDeliveryPerformanceParameters(dfDeliveryPerformanceParameters):
    try:
        # Get the latest data for material/combo appearing more than once.
        dfDeliveryPerformanceParameters = \
        dfDeliveryPerformanceParameters.groupby(['Material', \
                                                 'VendorCode']).agg({'Deviation_DeliveryDate':'mean', \
                                                                     'Deviation_DeliveredQty' : 'mean', \
                                                                     'Deviation_DeliveredValue':'mean', \
                                                                     'MaterialCountByVendor':'max' \
                                                                    }) \
                              .reindex(['Deviation_DeliveryDate', \
                                        'Deviation_DeliveredQty', \
                                        'Deviation_DeliveredValue', \
                                        'MaterialCountByVendor' \
                                       ], axis=1) \
                              .reset_index()
        dfDeliveryPerformanceParameters['Deviation_DeliveryDate'] = dfDeliveryPerformanceParameters['Deviation_DeliveryDate'].round(2)
        dfDeliveryPerformanceParameters['Deviation_DeliveredQty'] = dfDeliveryPerformanceParameters['Deviation_DeliveredQty'].round(2)
        dfDeliveryPerformanceParameters['Deviation_DeliveredValue'] = dfDeliveryPerformanceParameters['Deviation_DeliveredValue'].round(2)
        return  dfDeliveryPerformanceParameters
    except:
        s_classMethod       = 'Get the unique delivery performance parameters'
        s_statusType        = 'Error'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
def MergeVendorData(dfDataVendorMaxDate,dfDeliveryPerformanceParameters):
    try:
        s_classMethod       = 'Finalize the columns in dataframe.'
        s_statusType        = 'Success'    
        # Concat all the columns required for the analysis.
        dfDataVendorFinal = dfDataVendorMaxDate.merge(dfDeliveryPerformanceParameters,on=['Material','VendorCode'],how='inner')
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
def featureEngineerCategories_ApplyScoring(dfDataVendorFinal):
    try:
        s_classMethod       = 'Treat Categorical Varaibles'
        s_statusType        = 'Success' 
        # Build the dataframe with the encoded column for the categorical data.
        dfMaterialOHE = pd.get_dummies(dfDataVendorFinal['Material'])
        dfVendorOHE = pd.get_dummies(dfDataVendorFinal['VendorCode'])
        # Add the encoded columns to the existing dataframe.
        dfDataVendorFinal = pd.concat([dfDataVendorFinal,dfMaterialOHE,dfVendorOHE],axis=1)
        # prepare the rank data by hard coding the weights. 
        dfDataVendorFinal['Rank'] = ( (dfDataVendorFinal['Deviation_DeliveryDate']*0.4) + \
                                      (dfDataVendorFinal['Deviation_DeliveredQty']*0.4) + \
                                      (dfDataVendorFinal['MaterialCountByVendor']*0.2) 
                                    )
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
def PrepareData(dfDataVendorFinal):
    try:
        s_classMethod       = 'Prepare Data'
        s_statusType        = 'Success' 
    
        # Copy the dataframe.
        dfDataVendorML = dfDataVendorFinal.copy()
    
        # Remove features not useful for the modelling.
        featureDrop = ['Doc. Date'
                       ,'Material'
                       ,'VendorCode'
                       ,'UOM'
                       ,'Del Date'
                       ,'Sum of      Net Price'
                       ,'Sum of PO Quantity'
                       ,'Sum of PO Value'
                       ,'Deviation_DeliveredValue'
                       ,'Rank'
                       ,'MaterialCountByVendor'
                       ]
        dfDataVendorML.drop(featureDrop, axis=1,inplace=True)
        # Extract the independent variable.
        X_Input = dfDataVendorML.copy()
        return X_Input
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
def StandardizeData(dfDataVendorML):
    try:
        s_classMethod       = 'Standardize the data'
        s_statusType        = 'Success' 
        
        # Get the trained standardizer from the pickle.
        objType = 'TN'
        standardizer    = applConfig.GetTrainedObjectFromPickle(objType)
        X_unseen_stdzd  = applConfig.standardize_new_data(dfDataVendorML,
                                                          standardizer)
        return X_unseen_stdzd
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
def PredictOnNewData(X_unseen_stdzd):
    try:
        s_classMethod       = 'Perform Model classification'
        s_statusType        = 'Success' 
    
        # Get the trained model and calibrated model from the pickle.
        objType = 'TM'
        lr_optimal    = applConfig.GetTrainedObjectFromPickle(objType)
        objType = 'TCM'
        calibratedCCV = applConfig.GetTrainedObjectFromPickle(objType)
    
        # Get the predictions from the unseen set.
        Y_pred_unseen,Y_pred_proba_unseen = applConfig.GetPredictionsOnUnseenData( lr_optimal,
                                                                                   X_unseen_stdzd)
    
        # Get the calibrated predictions from the unseen set.
        Y_pred_calib_unseen = applConfig.GetCalibratedPredictionsOnUnseenData( calibratedCCV,
                                                                               X_unseen_stdzd)
        return Y_pred_unseen,Y_pred_proba_unseen,Y_pred_calib_unseen
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
def GetClassProbabilities(dfDataVendorFinal,Y_pred_calib_unseen):
    try:
        # Update class perdictions back to the main dataframe.
        # Class 0 ==> Has deviations.
        # Class 1 ==> Has no deviations.
        dfDataVendorFinal['class_Deviation_score'] = np.round(Y_pred_calib_unseen[:,0],3)
        # Update class probabilities back to the main dataframe.
        dfDataVendorFinal['class_NoDeviation_score'] = np.round(Y_pred_calib_unseen[:,1],3)
        
        # Compute Is_Deviation flag.
        dfDataVendorFinal['IsDeviation'] = 'Yes'
        dfDataVendorFinal.loc[( (dfDataVendorFinal['Deviation_DeliveryDate']   <= 0.0) & \
                                (dfDataVendorFinal['Deviation_DeliveredQty']   == 0.0) \
                              ),  \
                             'IsDeviation'] = 'No'
        return dfDataVendorFinal
    except:
        s_statusType        = 'Error'
        s_classMethod       = 'Update class probabilities'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription) 
#===================================================================================================
def GetMasterData(dfDataVendor):
    try:
        s_classMethod       = 'Extract Master Data'
        s_statusType        = 'Success'
        # Get the vendor name and code.
        dfDataVendor['VendorName'] = dfDataVendor['Supplier/Supplying Plant']
        # Extract the vendor name.
        dfDataVendor.VendorName.replace(to_replace=r'(\d+)', value='', regex=True,inplace=True)
        # Get distinct vendors.
        dfVendorMaster = dfDataVendor[['VendorCode','VendorName']].drop_duplicates()
        # Get distinct materials.
        dfMaterialMaster = dfDataVendor[['Material','Short Text']].drop_duplicates()
        return dfMaterialMaster,dfVendorMaster
    except:
        s_statusType        = 'Error'
        s_statusDescription = s_statusDescription + str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription)
#===================================================================================================
def ExportDataToCSV(dfDataVendorFinal,dfMaterialMaster,dfVendorMaster):
    try:
        # Set name for the dataframe's index.
        dfDataVendorFinal.index.name = "id"
        dfMaterialMaster.index.name = "id"
        dfVendorMaster.index.name = "id"
        '''
        ===== MATERIAL ========================================================
        '''
        # Remove trailing spaces.
        dfMaterialMaster['Material'] = dfMaterialMaster['Material'].str.strip()
        dfMaterialMaster['Short Text'] = dfMaterialMaster['Short Text'].str.strip()
        
        # Rename Material master columns as per Vendor portal requirements.
        dfMaterialMaster.rename(columns = {"Material": "material", 
                                           "Short Text": "short_Text"},
                                inplace=True
                               )
        # Build the path to store the Material Master file.
        opMatMasterFile = applConfig.Output + 'material.csv'
        dfMaterialMaster[['material', 'short_Text']].to_csv(opMatMasterFile)

        '''
        ===== VENDOR ==========================================================
        '''
        # Remove trailing spaces.
        dfVendorMaster['VendorCode'] = dfVendorMaster['VendorCode'].str.strip()
        dfVendorMaster['VendorName'] = dfVendorMaster['VendorName'].str.strip()
        # Rename Vendor master columns as per Vendor portal requirements.
        dfVendorMaster.rename(columns = {"VendorCode": "code", 
                                         "VendorName": "name"},
                                inplace=True
                               )
        # Build the path to store the Vendor Master file.
        opVendorMasterFile = applConfig.Output + 'vendor.csv'
        dfVendorMaster[['code', 'name']].to_csv(opVendorMasterFile)
        
        '''
        ===== VENDOR RANKING ==================================================
        '''
        # Rename VendorRanking columns as per Vendor portal requirements.
        dfDataVendorFinal.rename(columns = {"Material": "material", 
                                           "VendorCode": "vendorCode",
                                           "UOM":"uom",
                                           "Doc. Date" : "doc_Date",
                                           "Del Date":"del_Date",
                                           "Sum of PO Quantity":"sum_of_PO_Quantity",
                                           "Sum of      Net Price":"sum_of_Net_Price",
                                           "Sum of PO Value":"sum_of_PO_Value",
                                           "MaterialCountByVendor":"materialCountByVendor",
                                           "Rank":"rank",
                                           "Deviation_DeliveryDate":"deviation_DeliveryDate",
                                           "Deviation_DeliveredQty":"deviation_DeliveredQty",
                                           "IsDeviation":"isDeviation"
                                           },
                                inplace=True
                               )
        
        # Get the path to store the result.
        outputFile = applConfig.Output + 'VendorRanking.csv'

        # Export it to csv format.
        dfDataVendorFinal[['material','vendorCode','uom','doc_Date','del_Date' \
                           ,'sum_of_PO_Quantity','sum_of_Net_Price','sum_of_PO_Value','materialCountByVendor','rank' \
                           ,'deviation_DeliveryDate', 'deviation_DeliveredQty' \
                           ,'class_NoDeviation_score','class_Deviation_score' \
                           ,'isDeviation']].to_csv(outputFile)
        return outputFile
    except:
        s_statusType        = 'Error'
        s_classMethod       = 'Export data to csv'
        s_statusDescription = str(sys.exc_info())
        applConfig.WriteToActivityLog(s_classApplication,
                                      s_classMethod,
                                      s_statusType,
                                      s_statusDescription
                                     )    
        raise Exception(s_statusDescription) 
#===================================================================================================
#===================================================================================================
# Call the functions when this python file is run directly.
if __name__ == "__main__":
    # Instantiate the ML class.
    instantiateMLPipleline()
    # Read data from csv.
    dfCV = read_fromcsv()
    # Check for exceptions in data.
    GetAllExceptions(dfCV)
    # Get unique Material/Vendor combo.
    dfDataVendorMaxDate = RollupMultipleMaterialVendorCombo(dfCV)
    # Get delivery performance parameters.
    dfDeliveryPerformanceParameters = GetDeliveryPerformanceParameters()
    # Get unique delivery performance parameters.
    dfDeliveryPerformanceParametersUnique = GetUniqueDeliveryPerformanceParameters(dfDeliveryPerformanceParameters)
    # Merge Vendor Data.
    dfDataVendorFinal = MergeVendorData(dfDataVendorMaxDate,dfDeliveryPerformanceParametersUnique)
    # Feature engineer categories.
    dfDataVendorFinal = featureEngineerCategories_ApplyScoring(dfDataVendorFinal)
    # Prepare Data.
    X_Input = PrepareData(dfDataVendorFinal)
    # Standardize the data.
    X_unseen_stdzd = StandardizeData(X_Input)
    # Predict On New Data.
    Y_pred_unseen,Y_pred_proba_unseen,Y_pred_calib_unseen = PredictOnNewData(X_unseen_stdzd)
    # Update class probabilities into the main dataframe.
    dfDataVendorFinal = GetClassProbabilities(dfDataVendorFinal,Y_pred_calib_unseen)
    # Get Material master and Vendor master data.
    dfMaterialMaster,dfVendorMaster = GetMasterData(dfCV)
    # Export the data to csv.
    outputFile = ExportDataToCSV(dfDataVendorFinal,dfMaterialMaster,dfVendorMaster)
    print("Ranking of vendors completed successfully !!!!")
    