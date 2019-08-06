#======= Import the neccessary modules ===================================
import os
import json
import sys
import uuid
import linecache
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


#=========================================================================
# Class definition
class modMLModelingPipeline:
    # Instantiate all variables and methods required for the class.
    def __init__(self,ClientID,ClientName,Source,Output,
                 TrainedModel,ExecutionLog,ExecutionLogFileName,Archive,
                 dfX,dfY,total_rowCount,IsTimeBasedSplitting,Training_split_ratio,
                 Crossvalidation_split_ratio, Test_split_ratio
                ):
        #====== Set all the App config parameters from the json, input and output data ======
        self.ClientID                    = ClientID
        self.ClientName                  = ClientName
        self.Source                      = Source
        self.Output                      = Output
        self.TrainedModel                = TrainedModel
        self.ExecutionLog                = ExecutionLog
        self.ExecutionLogFileName        = ExecutionLogFileName
        self.Archive                     = Archive
        self.dfX                         = dfX
        self.dfY                         = dfY
        self.total_rowCount              = total_rowCount
        self.IsTimeBasedSplitting        = IsTimeBasedSplitting
        self.Training_split_ratio        = Training_split_ratio
        self.Crossvalidation_split_ratio = Crossvalidation_split_ratio
        self.Test_split_ratio            = Test_split_ratio
        #====================================================================================

    # Split the input and output variables based on specified splitting ratios.
    def split_data(self):
        #====== Variables to hold the split data ====================
        global X_train
        global X_cv
        global X_test
        global Y_train
        global Y_cv
        global Y_test
        #===========================================================
        # Check, if timebased splitting is required.
        if self.IsTimeBasedSplitting == 1:
            #Do time based splitting.
            # Split the training set.
            trainingUBound = round(self.Training_split_ratio * self.total_rowCount)
            X_train = self.dfX[0:trainingUBound]
            Y_train = self.dfY[0:trainingUBound]
            # Split the cross-validation set.
            crossvalidationLBound = trainingUBound
            crossvalidationUBound = crossvalidationLBound + round(self.Crossvalidation_split_ratio * self.total_rowCount)
            X_cv = self.dfX[crossvalidationLBound:crossvalidationUBound]
            Y_cv = self.dfY[crossvalidationLBound:crossvalidationUBound]
            # Split the test set.
            testLBound = crossvalidationUBound
            testUBound = testLBound + round(self.Test_split_ratio * self.total_rowCount)
            X_test = self.dfX[testLBound:testUBound]
            Y_test = self.dfY[testLBound:testUBound]
        else:
            # Do random splitting.
            cv_size   = (2 * self.Crossvalidation_split_ratio) # 0.4
            test_size = Test_split_ratio                       # 0.2
            X_train ,X_cv = train_test_split(self.dfX,test_size=cv_size) 
            Y_train ,Y_cv = train_test_split(self.dfY,test_size=cv_size) 
            X_cv ,X_test  = train_test_split(X_cv,test_size=test_size) 
            Y_cv ,Y_test  = train_test_split(Y_cv,test_size=test_size)

        # Return the split data.
        return X_train,X_cv,X_test,Y_train,Y_cv,Y_test
        
    # Standardize or Normalize the data.
    def standardize_data(self,IsNormalize=0):
        global X_train_stdzd
        global X_cv_stdzd
        global X_test_stdzd
        global Y_train_ravel
        global Y_cv_ravel
        global Y_test_ravel
        
        # Check, if data is to be normalized/standardized.
        if IsNormalize == 1:
            # Instantiate the normalizer
            normalizer = Normalizer()
        else:
            # Instantiate the standardizer
            normalizer = StandardScaler()
            
        #Standardize the data.
        X_train_stdzd = normalizer.fit_transform(X_train)
        X_cv_stdzd    = normalizer.transform(X_cv)
        X_test_stdzd  = normalizer.transform(X_test)
        
        #Flatten the Y into 1D array.
        Y_train_ravel = np.ravel(Y_train, order = 'C') 
        Y_cv_ravel    = np.ravel(Y_cv, order = 'C') 
        Y_test_ravel  = np.ravel(Y_test, order = 'C') 
        
        # Return the standardized data.
        return X_train_stdzd,X_cv_stdzd,X_test_stdzd,Y_train_ravel,Y_cv_ravel,Y_test_ravel
        

    # Method to return the non-calibrated model after training.
    def GetTrainedModel(self,optimalHyperparameter):
        # instantiate the Logistic Regression model with the optimal lambda.
        global lr_optimal
        lr_optimal = LogisticRegression(penalty='l2',              # use L2 regularizer.
                                        C=(optimalHyperparameter), # use Inverse of Lambda.
                                        class_weight='balanced',   # uses the values of y to automatically adjust 
                                                                   # weights in the input 
                                                                   # data, since we have class imbalance.
                                        solver='liblinear')        # solver = 'liblinear' because of low dimension and 
                                                                   # L2 regularizer.

        lr_optimal.fit(X_train_stdzd, Y_train_ravel)     # fitting the Logistic Regression model.
        return lr_optimal
        
    # Method to return the calibrated model after training.
    def GetCalibratedModel(self):
        ### Use CalibratedClassifierCV
        # Instantiate the CalibratedClassifierCV model.
        global calibratedCCV
        calibratedCCV = CalibratedClassifierCV(lr_optimal,       # Logistic Regression model.
                                               method='sigmoid', # sigmoid function.
                                               cv=5)             # crossvalidation #'s.
    
        calibratedCCV.fit(X_train_stdzd, Y_train_ravel)          # fit the model on the training set.
        return calibratedCCV

    # Method to get predictions.
    def GetPredictions(self):
        global Y_pred_test
        Y_pred_test = lr_optimal.predict(X_test_stdzd)   # predict,response from the Logistic Regression model.
        return X_test_stdzd, Y_pred_test
    
    # Method to get calibrated predictons.
    def GetCalibratedPredictions(self):
        global Y_pred_calib
        Y_pred_calib = calibratedCCV.predict_proba(X_test_stdzd)[:, 1]   # predict class probabilities on the testset.
        return X_test_stdzd, Y_pred_calib
    
    # Method to get the classifier metrics.
    def GetModelConfusionMatrix(self):
        # Plot the confusion matrix for the trained model.
        conf_mat = confusion_matrix(Y_test, Y_pred_test)
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_mat, annot=True, fmt='d'
                    ,xticklabels=['Not Selected','Selected'], yticklabels=['Not Selected','Selected'])
        plt.title('Confusion Matrix :')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return plt

    # Method to get the classifier ROC.
    def GetModelROC(self):
        # Plot the ROC curve for the trained model.
        # Calculate the RoC curve
        fpr, tpr, thresholds = roc_curve(Y_test_ravel,Y_pred_calib,pos_label=1)   # compute the roc curve.
        plt.plot([0, 1], [0, 1], linestyle='--')                                  # plot the 50% probability line.
        plt.plot(fpr, tpr, marker='.')                                            # plot the roc curve for the model.
        plt.title("ROC Curve.")                                                   # set the title of the plot.
        plt.xlabel("False Positive Rate")                                         # set the x label of the plot.
        plt.ylabel("True Positive Rate")                                          # set the y label of the plot.
        return plt

#===== Move this outside of this class ====================================================================================
    # Method to get predictions.
    def GetPredictionsOnUnseenData(self):
        # predict,response from the Logistic Regression model.
        global Y_pred_unlabelled
        Y_pred_unlabelled = lr_optimal.predict(X_unlabelled_stdzd)
        return X_unlabelled_stdzd, Y_pred_unlabelled
    
    # Method to get calibrated predictons.
    def GetCalibratedPredictionsOnUnseenData(self):
        # predict class probabilities on the unlabelledset.
        global Y_pred_unlabelled_calib
        Y_pred_unlabelled_calib = calibratedCCV.predict_proba(X_unlabelled_stdzd)[:, 1] 
        return X_unlabelled_stdzd, Y_pred_unlabelled_calib
#==========================================================================================================================
    
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

        with open(executionLog + 'ExecutionLog.txt', 'w+') as outfile:
            json.dump(data, outfile)
        #====================================================================
        
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
        if statusType != 'Success':
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