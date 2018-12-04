READMe File for Airbus Ship detection Challenge
@author: Shruti Agrawal
	 Maleeha Shabeer Koul
	 Mahasweta Sarma
	 Abhisek Banerjee


Following is the information required to understand this Ship detection project and be able to run the code and get results.

1>Python version used:

	3.5.6


2>Required headers:

    Following headers are required to run the python files. So all of them need to be installed so that the programs can run.
	1> cv2
	2> numpy
	3> imageio
	4> os
	5> imageio.core.util
	6> glob
	7> pandas
	8> print_function from __future__
	9> pyplot from matplotlib
	10> datasets from sklearn
	11> train_test_split from sklearn.model_selection
	12> GridSearchCV from sklearn.model_selection
	13> classification_report from sklearn.metrics
	14> confusion_matrix from sklearn.metrics
	15> accuracy_score from sklearn.metrics
        16> AdaBoostClassifier from sklearn.ensemble	
	17> XGBClassifier from xgboost
	18> model_selection from sklearn
	19> preprocessing from sklearn
        20> StandardScaler from sklearn.preprocessing
        21> warnings
	22> joblib from sklearn.externals
	23> SVC from sklearn.svm
	24> MLPClassifier from sklearn.neural_network
	25> RandomForestClassifier from sklearn.ensemble
	26> xgboost
	27> display from IPython.display
	28> pickle 
	29> roc_curve from sklearn.metrics
	30> roc_auc_score
  from sklearn.metrics

3> Input:

    None of the program files requires and command line argument. Below is a list of all the program files,
	1> dataset_filter.py
	2> crop.py
	3> hog.py
	4> eval_ada_xgboost.py
	5> eval_svm_mlp.py
	6> recognition_module.py

These files need to be placed in the same directory. Apart from these python files, there are few more files and directories/folders that need to be created/placed in the same directory.
    Folders/Directories:
	1> train_dataset
	2> train
	3> test
	4> train_pos
	5> train_neg
	6> test_pos
	7> test_neg
	8> demo (included in the submission)

     Files:
	1> train_ship_segmentations_v2.cv (Can be found at https://www.kaggle.com/c/airbus-ship-detection/data)

Usage and description of all these entries are described in the next section.


4> Details of files and folders, how to run, and output:

Step 1. Get the train_v2.zip folder from https://www.kaggle.com/c/airbus-ship-detection/data. Extract its contents(image files) and store it in train_dataset.

Step 2. Now run the dataset_filter.py file. This file reads image files from the folder train_dataset and puts all the positive images into the folder positive_data_samples(this folder gets created by the program itself) using the information provided in the csv file 'train_ship_segmentations_v2.cv. The contents of the folder positive_data_samples are then split into training and test data sets. We manually picked a subset of images, split them into training and test samples in a 7:3 ratio and stored them in train and test folders.

Step 3. Now run the crop.py file. This file reads images from train and test folders, crops them and puts the cropped images in train_cropped and test_cropped folders (folders are created by the program itself) respectively. We manually separated positive/negative examples of train_cropped and test_cropped and put them into train_pos, train_neg, test_pos, test_neg.

Step 4. Now run the hog.py file. This will read images from train_pos, train_neg, test_pos, test_neg and creates 2 csv files, train_data.csv and test_data.csv. Each csv file contains 2916 feature values as columns and one classification column (0 means positive/ 1 means negative).

Step 5 : Now run two files eval_ada_xgboost.py and eval_svm_mlp.py. These two files will run grid search on classifiers such as AdaBoost, XGBoost, SVM and MLP, find out the best parameters and create models with them. 4 files will be created after these files are run, 'SupportVectorMachine.pkl', 'Addaboost Classifier.pkl', 'XGBoost.pkl', 'NeuralNetwork.pkl', which store the model. Also, the python files generate accuracy, confusion matrix, and ROC curve for each classifier.

Step 6: Put some test files in the demo directory and run reognition_module.py to get the visual representation of the predictions on the image. We have chosen XGboost here since that has produced the best result. To test the accuracy of the models, please use the uploaded .pkl files.
