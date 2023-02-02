import numpy as np
import pandas as pd 
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import calendar
import en_core_web_sm

from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC     ### SVM for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

import random
import numpy as np
import math
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
# instantiate labelencoder object
from sklearn.tree import export_text

from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from datetime import datetime

nlp = spacy.load("en_core_web_lg")

genome_scores = pd.read_csv("D:/Document/Recommended_system/Recommended_systems/ml-25m/genome-scores.csv")

genome_tags = pd.read_csv("D:/Document/Recommended_system/Recommended_systems/ml-25m/genome-tags.csv")


tags = pd.read_csv("D:/Document/Recommended_system/Recommended_systems/ml-25m/tags.csv")

ratings = pd.read_csv("D:/Document/Recommended_system/Recommended_systems/ml-25m/ratings.csv")

movies  = pd.read_csv("D:/Document/Recommended_system/Recommended_systems/ml-25m/movies.csv")
rating_0_1  = ratings.iloc[:,2].values

rating_0_1[rating_0_1 <4]  = 0
rating_0_1[rating_0_1 >=4]  = 1 

def create_user(): 
    
# lay tung user - 14592 user
    userid_unique = tags.loc[:,"userId"].unique()
    
    user_list_tags = []   # user list  ma co dieu kien la all tag phai lon hon 200
    for user in userid_unique:
        
        n_user = np.where(tags.loc[:,"userId"] ==user)[0]
        
        #if ( len(n_user) >9  and len(n_user) <16 ):
        if ( len(n_user) > 200):   # thay doi user list neu muon
            print(user)
            user_list_tags.append(user)
    
    user_list_ratting_tags = []
    for i in user_list_tags :
        index_user = np.where(ratings.loc[:,"userId"] == i)[0]
        
        ratings_1  = np.where(ratings.loc[index_user,"rating"] == 1)[0]   # những thằng user_1 và có rating là 1 -> thích
        
        ratings_0  = np.where(ratings.loc[index_user,"rating"] == 0)[0]   # những thằng user_1 và có rating là 0 -> không thích 
          
        if (len(ratings_1 ) >= 40    and len(ratings_0 ) >=40  ):
            print(i)
            user_list_ratting_tags.append(i)
            
    return user_list_ratting_tags

user_list_ratting_tags = create_user()
        


    
    
    
def data_user():
    listt = []
    for i in user_list_ratting_tags:
        
        df_user = pd.read_csv(f"D:/Document/Recommended_system/Recommended_systems/ml-25m/data_tu_15_20tag/data/data_user({i}).csv")
        
        
        listt.append(df_user)  
    
    data_user = pd.concat(listt)
    data_user = data_user.reset_index()
    
    del data_user['Unnamed: 0']
    del data_user['index']
    data_user = data_user.rename(columns={"rating":"Class"})
    
    return data_user

data_user = data_user()


def del_columns(data_user):
    
    del data_user["Year"]
    del data_user["movieId"]
    del data_user["timestamp"]
    del data_user["Tags"]
    del data_user["unknown"]
    del data_user["Month"]
    del data_user["title"]
    del data_user["Weekday"]
    del data_user["genres"]
    #del data_user["max_similarity_genome_with_user"]
    
    
    
del_columns(data_user)
    

def each_user(data_user):

    r,c = data_user.shape
    acc_DT =list()
    acc_SVM =list()
    acc_Navie = list()
    acc_Bagging = list()
    acc_RF = list()
    acc_BT  = list()
    
    time_DT =list()
    time_SVM =list()
    time_Naive =list()
    time_Bagging =list()
    time_RF =list()
    time_BT = list()
    
    for id_user in list_user_oke:

        
         index = np.where(data_user.loc[:,"userId"]  == id_user)[0]
         
         data_user_official =data_user.loc[index,:]
         del data_user_official["userId"]
         
         for i in range(0,20):
             
            Y = data_user_official['Class']
            X = data_user_official.iloc[:,data_user_official.columns !='Class']
            
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,  test_size=0.2, random_state = random.randint(1,100000))
        
            
 
     
            # decision tree        
            start = datetime.now()
    
            model = DecisionTreeClassifier()        
            model.fit(X_Train, Y_Train)            
            predictions = model.predict(X_Test)
            end = datetime.now() -start
            time_DT.append(end)
    
            print("Accuracy Cay Quyet Dinh:",metrics.accuracy_score(Y_Test, predictions))
            acc_DT.append(metrics.accuracy_score(Y_Test, predictions))
                

            #cm = metrics.confusion_matrix(Y_Test, predictions ,labels = model.classes_)
            #dsp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= model.classes_)
            #dsp.plot()
            
     
    
    # SVM
            
            start = datetime.now()
                
            svclassifier = SVC(kernel='rbf')
            svclassifier.fit(X_Train, Y_Train.values.ravel())
            y_pred = svclassifier.predict(X_Test)
            end = datetime.now() -start
            time_SVM.append(end)
    
            print("Accuracy SVM:",metrics.accuracy_score(Y_Test, y_pred))
            acc_SVM.append(metrics.accuracy_score(Y_Test, y_pred))
            
    
    # naive classifier
            X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,  train_size=0.7, random_state = random.randint(0,100000))
            start = datetime.now()
            model_navie = GaussianNB()
          #  model_navie = MultinomialNB()        
            model_navie.fit(X_Train, Y_Train.values.ravel()) 
            prediction = model_navie.predict(X_Test) 
            end = datetime.now() -start
            time_Naive.append(end)
            print("Naive Bayes: ", metrics.accuracy_score(Y_Test, prediction))
            acc_Navie.append(metrics.accuracy_score(Y_Test, prediction))
           
        
    # bagging
           # base_cls = SVC()
            #base_cls = KNeighborsClassifier(n_neighbors=5)
            start = datetime.now()
    
            base_cls = DecisionTreeClassifier()
            
            #model = BaggingClassifier(base_estimator = base_cls, n_estimators = 100)
            model_bagging = BaggingClassifier(base_estimator = base_cls, n_estimators = 100)
            
            model_bagging.fit(X_Train, Y_Train.values.ravel())
            predictions = model_bagging.predict(X_Test)
            
            end = datetime.now() -start
            time_Bagging.append(end)
                    
            acc = metrics.accuracy_score(Y_Test, predictions)    
            acc_Bagging.append(acc)        
            print ("Bagging: ",acc)     
            
            
   # boosting
            
            start = datetime.now()
            model_boosting = GradientBoostingClassifier(n_estimators=100)
            
            
            model_boosting.fit(X_Train ,Y_Train.values.ravel())
            prediction_boosting = model_boosting.predict(X_Test)
            
            end = datetime.now() -start
            time_BT.append(end)
            
            accBT = metrics.accuracy_score(Y_Test, prediction_boosting)
            acc_BT.append(accBT)
            
            print ("boosting: ",accBT)   

            
    #random forest
            start = datetime.now()
    
            rf_model = RandomForestClassifier(n_estimators=100, max_features= 6)
            rf_model.fit(X_Train,Y_Train.values.ravel())
            pred_y = rf_model.predict(X_Test)
               
            end = datetime.now() -start
            time_RF.append(end)
                    
            accRF = metrics.accuracy_score(Y_Test, pred_y)
            acc_RF.append(accRF)
            print ("Accuracy RF: ",accRF) 
        
            
    
    results =[]
    results.append(acc_DT)
    results.append(acc_SVM)
    results.append(acc_Navie)
    results.append(acc_Bagging)
    results.append(acc_RF)
    results.append(acc_BT)
    # results.append(acc_RandomForest)
    # results.append(acc_Boosting)
    # results.append(acc_Stacking)


    #names =('decision tree','randomforest','boosting','stacking')
    names =('Decision tree', 'SVM', 'Navie bayes','Bagging','Random forest' ,'Boosting')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    
   # ax = fig.add_subplot(111)
#    plt.boxplot(results)
    #plt.boxplot(results, labels=names, showmeans = True)
    

    plt.boxplot(results, labels=names)
    
    plt.xticks( rotation=40, fontsize=10)
    plt.ylabel('Accuracy')    

    #ax.set_xticklabels(names)
    plt.show()            
    
    print ("Results")
    print("\n")
    print ("decision tree")
    print (np.mean(acc_DT))    ## gia tri trung binh  
    print (np.std(acc_DT))     ## Do lech chuan   # mức độ dàn trải của dữ liệu
                                # càng gần trung bình thì càng nhỏ 
    
    
    print ("SVM")
    print (np.mean(acc_SVM))    ## gia tri trung binh  
    print (np.std(acc_SVM))     ## Do lech chuan   
    
    print ("Naive Bayes")
    print (np.mean(acc_Navie))    ## gia tri trung binh  
    print (np.std(acc_Navie))     ## Do lech chuan   
    
    print ("Bagging")
    print (np.mean(acc_Bagging))    ## gia tri trung binh  
    print (np.std(acc_Bagging))     ## Do lech chuan   
    
    print ("Boosting")
    print (np.mean(acc_BT))    ## gia tri trung binh  
    print (np.std(acc_BT))     ## Do lech chuan   
    
    print ("Random forest")
    print (np.mean(acc_RF))    ## gia tri trung binh  
    print (np.std(acc_RF))     ## Do lech chuan   
    
    print ("Time")
    print ("Dt:",np.mean(time_DT))
    print ('SVM',np.mean(time_SVM))
    print ('NAive',np.mean(time_Naive))
    print ('Bagging',np.mean(time_Bagging))
    print('Bt',np.mean(time_BT))
    print ('RAndom',np.mean(time_RF))
    
    
each_user = each_user(data_user)
    
