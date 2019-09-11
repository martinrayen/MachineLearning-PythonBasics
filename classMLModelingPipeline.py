#=========================================================================
# Class to hold the ML modelling methods.
# Author : Joseph MTV;
#=========================================================================

#======= Import the neccessary modules ===================================
import os
import sys
import uuid
import json
import pickle
import sklearn
import linecache
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
#=========================================================================

# Class definition
class modMLModelingPipeline:
    def __init__(self,ClientID,ClientName,Source,Output,
                 TrainedModel,ExecutionLog,ExecutionLogFileName,FeedActiveSheet,FeedSkipRows,Archive
                ):
        #====== Set all the App config parameters from the json, input and output data ======
        self.ClientID                    = ClientID
        self.ClientName                  = ClientName
        self.Source                      = Source
        self.Output                      = Output
        self.TrainedModel                = TrainedModel
        self.ExecutionLog                = ExecutionLog
        self.ExecutionLogFileName        = ExecutionLogFileName
        self.FeedActiveSheet             = FeedActiveSheet
        self.FeedSkipRows                = FeedSkipRows
        self.Archive                     = Archive
        #====================================================================================

    # Split the input and output variables based on specified splitting ratios.
    def split_data(self,
                   dfX,
                   dfY,
                   total_rowCount,
                   IsTimeBasedSplitting,
                   Training_split_ratio,
                   Crossvalidation_split_ratio,
                   Test_split_ratio                   
                  ):
        # Check, if timebased splitting is required.
        if IsTimeBasedSplitting:
            #Do time based splitting.
            # Split the training set.
            trainingUBound = round(Training_split_ratio * total_rowCount)
            X_train = dfX[0:trainingUBound]
            Y_train = dfY[0:trainingUBound]
            # Split the cross-validation set.
            crossvalidationLBound = trainingUBound
            crossvalidationUBound = crossvalidationLBound + round(Crossvalidation_split_ratio * total_rowCount)
            X_cv = dfX[crossvalidationLBound:crossvalidationUBound]
            Y_cv = dfY[crossvalidationLBound:crossvalidationUBound]
            # Split the test set.
            testLBound = crossvalidationUBound
            testUBound = testLBound + round(Test_split_ratio * total_rowCount)
            X_test = dfX[testLBound:testUBound]
            Y_test = dfY[testLBound:testUBound]
        else:
            # Do random splitting.
            # Split the data in the proportion, 0.6,0.2,0.2
            cv_size   = (2 * Crossvalidation_split_ratio)
            test_size = 0.5    
            # Issue here. Input data to this method should be inclusive of X and Y.
            X_train ,X_cv_i,Y_train ,Y_cv_i = train_test_split(dfX,dfY,test_size=cv_size) 
            #Y_train ,Y_cv_i = train_test_split(dfY,test_size=cv_size) 
            X_cv ,X_test,Y_cv ,Y_test = train_test_split(X_cv_i,Y_cv_i,test_size=test_size) 
            #Y_cv ,Y_test    = train_test_split(Y_cv_i,test_size=test_size)

        # Return the split data.
        return X_train,X_cv,X_test,Y_train,Y_cv,Y_test
        
  
    # Standardize or Normalize the data.
    def standardize_data(self,
                         X_train,
                         X_cv,
                         X_test,
                         Y_train,
                         Y_cv,
                         Y_test,
                         IsNormalize=0,
                         SkipNormalize=0
                        ):
        if not(SkipNormalize):
            # Check, if data is to be normalized/standardized.
            if IsNormalize:
                # Instantiate the normalizer
                normalizer = Normalizer()
            else:
                # Instantiate the standardizer
                normalizer = StandardScaler()
            #Standardize the data.
            X_train_stdzd = normalizer.fit_transform(X_train)
            X_cv_stdzd    = normalizer.transform(X_cv)
            X_test_stdzd  = normalizer.transform(X_test)
        else:
            # Do not standardize the data.
            X_train_stdzd = X_train.values
            X_cv_stdzd    = X_cv.values
            X_test_stdzd  = X_test.values
            # When in doubt, use standardization.
            normalizer = StandardScaler()
        
        #Flatten the Y into 1D array.
        Y_train_ravel = np.ravel(Y_train, order = 'C') 
        Y_cv_ravel    = np.ravel(Y_cv, order = 'C') 
        Y_test_ravel  = np.ravel(Y_test, order = 'C') 
        
        # Return the standardized data.
        return normalizer,X_train_stdzd,X_cv_stdzd,X_test_stdzd,Y_train_ravel,Y_cv_ravel,Y_test_ravel

    # Standardize or Normalize the data.
    def standardize_new_data(self,
                             X_new,
                             pklStandardizer,
                             SkipNormalize=0
                            ):
        if not(SkipNormalize):
            normalizer = pklStandardizer
            #Standardize the data.
            X_new_stdzd = normalizer.fit_transform(X_new)
        else:
            X_new_stdzd = X_new.values
        
        # Return the standardized data.
        return X_new_stdzd
    
#================= Decision Tree Algorithm ==============================================================================================
    # Method to get the optimal Hyper-parameters max_depth and min_sample_split in DT case.
    def  GetDTreeModelHyperParameters(self,
                                     X_train_stdzd,
                                     Y_train_ravel,
                                     classWeight = 0):
        # set the tree parameters.
        parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}

        if classWeight != 0:
            # instantiate the tree model.
            model=tree.DecisionTreeClassifier(class_weight='balanced')
        else:
            # instantiate the tree model.
            model=tree.DecisionTreeClassifier(class_weight=None)

        # intantitate the CrossValidation method.
        grid = GridSearchCV(estimator=model, param_grid=parameters)
        # fit the method.
        grid.fit(X_train_stdzd, Y_train_ravel)
        # summarize the results of the grid search
        gridResults = grid
        bestScore   = grid.best_score_
        optimal_HyperParameter = grid.best_params_

        # Return the optimal Hyper-parameter.
        return gridResults,bestScore,optimal_HyperParameter    

    # Method to return the non-calibrated model after training.
    def GetTrainedDTreeModel(self,
                            X_train_stdzd,
                            Y_train_ravel,                        
                            optimal_depthtree,
                            optimal_minsamplesplit,
                            classWeight=0
                           ):
        if classWeight != 0:
            print('classWt==balanced')
            # Instantiate the DecisionTree model with optimal hyper-paramters.
            dt_optimal = tree.DecisionTreeClassifier(criterion='gini',  # metric to measure quality of the split.
                                                     splitter = 'best', # strategy used to split at each node.
                                                     max_depth=(optimal_depthtree), # Hyper-parameter, "Depth" 
                                                                                    # of the current iteration.
                                                     min_samples_split = (optimal_minsamplesplit), # Hyper-parameter,
                                                                                                   #"minimum sample
                                                                                                   # split" of the
                                                                                                   # current iteration.
                                                     class_weight='balanced') # uses the values of y to automatically 
                                                                              # adjust weights in the input data.
        else:
            # Instantiate the DecisionTree model with optimal hyper-paramters.
            dt_optimal = tree.DecisionTreeClassifier(criterion='gini',  # metric to measure quality of the split.
                                                     splitter = 'best', # strategy used to split at each node.
                                                     max_depth=(optimal_depthtree), # Hyper-parameter, "Depth" 
                                                                                    # of the current iteration.
                                                     min_samples_split = (optimal_minsamplesplit), # Hyper-parameter,
                                                                                                   #"minimum sample
                                                                                                   # split" of the
                                                                                                   # current iteration.
                                                     class_weight=None) # uses the values of y to automatically 
                                                                        # adjust weights in the input data.

        dt_optimal.fit(X_train_stdzd, Y_train_ravel)                     # fit the model on the training set.

        return dt_optimal    
#======================================================================================================================================    
    # Method to get the optimal Hyper-parameter lambda in LR case.
    def  GetModelHyperParameters(self,
                                 X_train_stdzd,
                                 Y_train_ravel,
                                 penalty = 'l2',
                                 cv=3, 
                                 classWeight = 0,
                                 solver = 'liblinear',
                                 max_iter = 1000
                               ):
        # prepare a range of alpha values to test.
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000,1000000,10000000] }
        if classWeight != 0:
            # create and fit a logistic regression model, testing each alpha
            model = LogisticRegression(penalty=penalty,class_weight='balanced',max_iter = max_iter)
        else:
            # create and fit a logistic regression model, testing each alpha
            model = LogisticRegression(penalty=penalty,solver=solver,max_iter = max_iter)
        grid = GridSearchCV(estimator=model, param_grid=param_grid,n_jobs=-1,cv=cv)
        grid.fit(X_train_stdzd, Y_train_ravel)
        # summarize the results of the grid search
        gridResults = grid
        bestScore   = grid.best_score_
        optimal_HyperParameter = grid.best_estimator_.C
        
        # Return the optimal Hyper-parameter.
        return gridResults,bestScore,optimal_HyperParameter

    
    # Method to return the non-calibrated model after training.
    def GetTrainedModel(self,
                        X_train_stdzd,
                        Y_train_ravel,                        
                        optimalHyperparameter,
                        penalty='l2',
                        solver='liblinear',
                        classWeight=0,
                        max_iter = 1000,
                        sampleWeight =None
                       ):
        if classWeight != 0:
            print('classWt==balanced')
            # instantiate the Logistic Regression model with the optimal lambda.
            lr_optimal = LogisticRegression(penalty=penalty,           # use L2 regularizer.
                                            max_iter = max_iter,       # maximum iterations for convergence.
                                            C=(optimalHyperparameter), # use Inverse of Lambda.
                                            class_weight='balanced',   # uses the values of y to automatically adjust 
                                                                       # weights in the input 
                                                                       # data, since we have class imbalance.
                                            solver=solver)             # solver = 'liblinear' because of low dimension and 
                                                                       # L2 regularizer.
        else:
            print('classWt==None')
            # instantiate the Logistic Regression model with the optimal lambda.
            lr_optimal = LogisticRegression(penalty=penalty,           # use L2 regularizer.
                                            max_iter = max_iter,       # maximum iterations for convergence.
                                            C=(optimalHyperparameter), # use Inverse of Lambda.
                                            class_weight=None,         # uses the values of y to automatically adjust 
                                                                       # weights in the input 
                                                                       # data, since we have class imbalance.
                                            solver=solver)             # solver = 'liblinear' because of low dimension and 
                                                                       # L2 regularizer.
            
        # fitting the Logistic Regression model.
        lr_optimal.fit(X_train_stdzd,Y_train_ravel,sample_weight = sampleWeight) # sampleWeight for each instance using KL Divergence.

        return lr_optimal
        
    # Method to return the calibrated model after training.
    def GetCalibratedModel(self,
                           lr_optimal,
                           X_train_stdzd,
                           Y_train_ravel                          
                          ):
        ### Use CalibratedClassifierCV
        # Instantiate the CalibratedClassifierCV model.
        calibratedCCV = CalibratedClassifierCV(lr_optimal,       # Base model.
                                               method='sigmoid', # sigmoid function.
                                               cv=3)             # crossvalidation #'s.
    
        calibratedCCV.fit(X_train_stdzd, Y_train_ravel)          # fit the model on the training set.
        return calibratedCCV

    # Method to get predictions.
    def GetPredictions(self,
                       lr_optimal,
                       X_test_stdzd):
        Y_pred_test = lr_optimal.predict(X_test_stdzd)   # predict,response from the Logistic Regression model.
        return Y_pred_test
    
    # Method to get calibrated predictons.
    def GetCalibratedPredictions(self,
                                 calibratedCCV,
                                X_test_stdzd):
        Y_pred_calib = calibratedCCV.predict_proba(X_test_stdzd)[:, 1]   # predict class probabilities on the testset.
        return Y_pred_calib
    
    # Method to get the classifier metrics.
    def GetModelConfusionMatrixForBinaryClass(self,
                                              Y_test,
                                              Y_pred_test
                                             ):
        #https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
        # Plot the confusion matrix for the trained model.
        conf_mat = confusion_matrix(Y_test, Y_pred_test)
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_test).ravel()
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_mat, annot=True, fmt='d')
                    #,xticklabels=['Not Selected','Selected'], yticklabels=['Not Selected','Selected'])
                    #,xticklabels=tickLabels, yticklabels=tickLabels)
        plt.title('Confusion Matrix :')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # For multi-class, cannot return the below. Explore alternate ways.
        return plt, conf_mat, tn, fp, fn, tp

    # Method to get the classifier ROC.
    def GetModelROCForBinaryClass(self,
                                  Y_test_ravel,
                                  Y_pred_calib):
        # Plot the ROC curve for the trained model.
        # Get the ROC score
        roc_score = roc_auc_score(Y_test_ravel,Y_pred_calib)
        # Calculate the RoC curve
        fpr, tpr, thresholds = roc_curve(Y_test_ravel,Y_pred_calib,pos_label=1)   # compute the roc curve.
        plt.plot([0, 1], [0, 1], linestyle='--')                                  # plot the 50% probability line.
        plt.plot(fpr, tpr, marker='.')                                            # plot the roc curve for the model.
        plt.title("ROC Curve.")                                                   # set the title of the plot.
        plt.xlabel("False Positive Rate")                                         # set the x label of the plot.
        plt.ylabel("True Positive Rate")                                          # set the y label of the plot.
        return plt, roc_score

    # Method to get predictions and class probabilities.
    def GetPredictionsOnUnseenData(self,
                                   lr_optimal,
                                   X_unlabelled_stdzd):
        # predict,response from the Logistic Regression model.
        Y_pred_unlabelled = lr_optimal.predict(X_unlabelled_stdzd)
        Y_pred_proba_unlabelled = lr_optimal.predict_proba(X_unlabelled_stdzd)
        return Y_pred_unlabelled,Y_pred_proba_unlabelled
    
    # Method to get calibrated predictons.
    def GetCalibratedPredictionsOnUnseenData(self,
                                             calibratedCCV,
                                             X_unlabelled_stdzd):
        # predict class probabilities on the unlabelledset.
        Y_pred_unlabelled_calib = calibratedCCV.predict_proba(X_unlabelled_stdzd)
        return Y_pred_unlabelled_calib
    
    # Create all the required App config directories.
    def CreateAppDirectories(self):
        #======= Extract app config directories from json. ==================
        sourceLoc       = self.Source
        outputLoc       = self.Output
        trainedModelLoc = self.TrainedModel
        executionLog    = self.ExecutionLog
        archiveLog      = self.Archive
        #====================================================================
        #====== Check and Create App Config Directories =====================
        # Check, if Source directory exists. Else, create new.
        if not(os.path.isdir(sourceLoc)):
            os.mkdir(sourceLoc)
        # Check, if Output directory exists. Else, create new.
        if not(os.path.isdir(outputLoc)):
            os.mkdir(outputLoc)
        # Check, if TrainedModel directory exists. Else, create new.
        if not(os.path.isdir(trainedModelLoc)):
            os.mkdir(trainedModelLoc)
        # Check, if ExecutionLog directory exists. Else, create new.
        if not(os.path.isdir(executionLog)):
            os.mkdir(executionLog)
        # Check, if Archive directory exists. Else, create new.
        if not(os.path.isdir(archiveLog)):
            os.mkdir(archiveLog)
        #====================================================================
        #===== Create the ExecutionLog File with a dummy entry ==============
        #===== Run, only if ExecutionLog does not exist =====================
        executionLogFile = executionLog + 'ExecutionLog.txt'
        if not os.path.isfile(executionLogFile):
            data = {}
            data['ExecutionLog'] = []
            data['ExecutionLog'].append({
                'ID'         :'ClientID',
                'ExecutionID':'Global_Unique_Identifier',
                'DateTime'   :'Current_Date',
                'Class'      :'Python class',
                'Method'     :'Python method ',
                'StatusType' :'Success|Error',
                'Description':'Success|Error',
                'User'       :'User Login'

            })

            with open(executionLogFile, 'w+') as outfile:
                json.dump(data, outfile)
        
        return sourceLoc,outputLoc,trainedModelLoc,archiveLog
        #====================================================================
        
    
    # Pickle the trained object.
    def PickleTrainedObject(self,objType,trainedObject):
        if objType == 'TM': # Trained Model
            # Generate the destination filename.
            pklTrainedObject = self.TrainedModel + 'trainedModel_' + self.ClientID
            loggedFile       = self.Archive      + 'trainedModel_' + self.ClientID
        elif objType == 'TN': # Trained Normalizer
            pklTrainedObject = self.TrainedModel + 'trainedStandardizer_' + self.ClientID
            loggedFile       = self.Archive      + 'trainedStandardizer_' + self.ClientID
        elif objType == 'TCM': # Trained Calibrated Model
            pklTrainedObject = self.TrainedModel + 'trainedCalibratedModel_' + self.ClientID
            loggedFile       = self.Archive      + 'trainedCalibratedModel_' + self.ClientID
        else:
            pklTrainedObject = None

        #===== Move trained file to Log before overwite  ==================
        # Move the file to the log, if exists already.
        if os.path.isfile(pklTrainedObject + '.pkl'):
            today = date.today()
            now = datetime.now()
            current_time = now.strftime("%H%M%S")
            current_date_time = str(today) + '_' + str(current_time)
            loggedFile = loggedFile + '_' + current_date_time
            os.rename(pklTrainedObject + '.pkl', loggedFile + '.pkl')    
        
        # Open the picklefile in binary mode.
        pklFile = open(pklTrainedObject + '.pkl', 'ab')
        # Dump the trainedModel to the pickle file.
        pickle.dump(trainedObject, pklFile) 
        # Close the pickle file.
        pklFile.close()
        
        return pklTrainedObject


    # Pickle dataframes.
    def PickleDataframes(self,dfPickle):
        # Path of the pickled dataframe.
        picklePath = self.TrainedModel + 'dfDeliveryPerformanceParameters_' + self.ClientID
        loggedFile = self.Archive + 'dfDeliveryPerformanceParameters_' + self.ClientID
        # Move the file to the log, if exists already.
        if os.path.isfile(picklePath + '.pkl'):
            # Get the current date.
            today = date.today()
            # Get the current time.
            now = datetime.now()
            current_time = now.strftime("%H%M%S")
            # Concatenate date and time.
            current_date_time = str(today) + '_' + str(current_time)
            # Build the destination log file name.
            loggedFile = loggedFile + '_' + current_date_time
            os.rename(picklePath + '.pkl', loggedFile + '.pkl')    
        # Pickle the dataframe.
        dfPickle.to_pickle(picklePath + '.pkl')
        return picklePath

    # Read dataframe from pickle.
    def ReadDataframeFromPickle(self):
        picklePath = self.TrainedModel + 'dfDeliveryPerformanceParameters_' + self.ClientID + '.pkl'
        dfDPP = pd.read_pickle(picklePath)
        return dfDPP

        
        
    # Get the trained object from Pickle file.
    def GetTrainedObjectFromPickle(self,objType):
        # Generate the destination filename.
        if objType == 'TM': # Trained Model
            pklTrainedObject = self.TrainedModel + 'trainedModel_' + self.ClientID 
        elif objType == 'TN': # Trained Normalizer
            pklTrainedObject = self.TrainedModel + 'trainedStandardizer_' + self.ClientID
        elif objType == 'TCM': # Trained Calibrated Model
            pklTrainedObject = self.TrainedModel + 'trainedCalibratedModel_' + self.ClientID
        elif objType == 'DPP': # Dataframe to store the delivery performance parameters.
            pklTrainedObject = self.TrainedModel + 'dfDeliveryPerformanceParameters_' + self.ClientID 
            print(pklTrainedObject)
        else:
            pklTrainedObject = None

        # Open the trained object file in binary mode.
        pklFile = open(pklTrainedObject + '.pkl', 'rb')
        # Load the trained model from pickle.
        trainedObject = pickle.load(pklFile)
        # Close the pickle file.
        pklFile.close()
        # Return the trained model.
        return trainedObject


    # Write to activity log.
    def WriteToActivityLog(self,classApplication,classMethod,statusType,statusDescription=''):
        #===== Get the activity details to log ==============================
        userlogin   = str(os.getlogin())
        currentdt   = str(datetime.now())
        guid        = str(uuid.uuid4())
        activityLog = self.ExecutionLog
        clientID    = self.ClientID
        #====================================================================
        
        #====== Extract the error details to be logged ======================
        if statusType == 'Error':
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            statusDescription = 'Error occurred at line :' + str(lineno) + '| Code:' + str(line.strip()) + '|Error Description :' + str(exc_obj)
        #====================================================================
        
        # Open and read the App Configuration using json.
        with open(activityLog + 'ExecutionLog.txt','r+') as json_file:
            # Load the App config details.
            data = json.load(json_file)

        # Append, the activity entry.
        data['ExecutionLog'].append({
            'ID'         : clientID,
            'ExecutionID': guid,
            'DateTime'   : currentdt,
            'Class'      : classApplication,
            'Method'     : classMethod,
            'StatusType' : statusType,
            'Description': statusDescription,
            'User'       : userlogin

        })

        # Write the activity log to file.
        with open(activityLog + 'ExecutionLog.txt', 'w+') as outfile:
            json.dump(data, outfile)        