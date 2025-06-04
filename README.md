# Cherry Leaf Classifier

# Table of Contents

## [Dataset Content](#dataset-content-1)
## [Business Requirements](#business-requirements-1)
- [Questions](#business-requirement-questions)
## [Hypothesis](#hypothesis-and-how-to-validate)
## [Dashboard Design](#dashboard-design-1)
- [Dashboard Expectations](#dashboard-expectations)
## [Bugs](#bugs-1)
## [Technology Used](#main-data-analysis-and-machine-learning-libraries)
## [Deployment](#deployment-1)
## [Credits](#credits-1)


## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Business Requirement Questions

#### What is the business objective requiring an ML Solution?

1. Conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
2. Predicting if a cherry leaf is healthy or contains powdery mildew.

#### Is the data available for the training of the model, if yes which type of data, if no how will you get the data?

- Data is available, there is a dataset containing 2104 healthy cherry leaf images and 2104 cherry leaves containing powdery mildew images.
- It will need compiling into a train, val and test set.

#### Does the customer need a dashboard or API endpoint?

- The customer will require a dashboard.
- There is a dashboard expectation list further down below.

#### What does success look like?

- The client considers a study showing how to visually differentiate a cherry leaf that is healthy from one that contains a powdery mildew.
- The client will be able to access a dashboard and be able to immediately check whether a cherry leaf is healthy or one that contains powdery mildew by uploading an image file of the cherry leaf.

#### Can you break down the project into Epics and User Stories?

- Information and Data Collection
- Data Cleaning, Preprocessing, Visualisation
- Model Training
- Model Fitting
- Optimization 
- Validation
- Dashboard Planning and Design
- Dashboard Development
- Dashboard Deployment and Release

#### Ethical or Privacy concerns?

- The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.

#### What level of prediction performance is needed? 

- The client has agreed upon a 97% level of prediction for the performance metric accuracy. 

#### What are the project inputs and intended outputs?

Inputs:

* Cherry Leaf Image Dataset provided by the client, containing: 
* 2104 healthy images 
* 2104 images powdery
* [Cherry Leaves Dataset](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
* Preprocessed Images of size either 50*50 or 100*100.
* Pixel values normalised by dividing by 255.0 to scale data for model training.
* Target labels/classes: 'healthy', 'podwery_mildew'

Outputs:

* A trained CNN-based deep learning model.
* Visualisation of image data, label distributions, model history and performance metrics.
* A final model fully that generalises well on unseen test data with an accuracy of 97% or higher.
* A deployed streamlit dashboard that allows users to upload an image of a cherry leaf and instantly detect whether it contains powdery mildew or it is healthy.

#### Does the data suggest a particular model?

- The data provided suggests a binary classification model, since there will be two classifications, Healthy or Infected with Mildew.
- This will be class 0 for Healthy and class 1 for Containing Mildew.

#### How will the customer benefit?

- The customer will be able to quickly identify cherry leaves containing mildew or healthy. 
- They will not release any unhealthy cherry leaves to the market with a product of compromised quality. 


## Hypothesis and how to validate?

- This project explores whether a convolutional neural network (CNN) based binary classification model can accurately classify cherry leaves as either healthy or ones containing powdery mildew. The model will be able to achieve a prediction accuracy of at least a **97%**.

### Null Hypothesis

A CNN model trained on cherry leaf images cannot reliably distinguish between healthy leaves and ones containing powdery mildew better than random chance (**50%** accuracy).

### Alternative Hypothesis

A CNN model trained on cherry leaf images can reliably distinguish between healthy leaves and ones containing powdery mildew with at least a **97%** accuracy.

## Dashboard Design

### Dashboard Expectations:

- A project summary page: showing the project dataset summary and the client's requirements.
- A page listing your findings related to a study to visually differentiate a cherry leaf that is healthy from one that contains a powdery mildew.
- A page containing:
- A link to download a set of cherry leave images for live prediction.
- A user interface with a file uploader widget, the user should be allowed to upload multiple images. Each image will be displayed along with a prediction statement indicating whether the cherry leaf is healthy or contains powdery mildew and the probability associated with this statement. 
- A table with the image name and prediction results, along with a download button to download the table.
- A page indicating your project hypothesis and how you validated it across the project. 
- A technical page displaying model performance.

## Dashboard Design

### Project Summary

This provides an overview of the project to the client, acknowledging the business case requirements, the main cause for the business case and the goals of the client. 

### Project Hypothesis

This provides the client with a general hypothesis, a null hypothesis and a alternative hypothesis. The final model performance is presented as evidence in favour of the alternative hypothesis and not in favour of the null hypothesis.

### Leaf Visualisation

This section provides the client with:
- Insight into image variability within individual classes. (Healthy and Powdery Mildew)
- Insight into image standard deviation between classes. (Healthy against Powder Mildew)
- It also provides an image montage of both healthy leaves and leaves infected with powdery mildew.
- It includes a radio button allowing the client to select a specific section of the image analysis. 

### ML Performance Metrics

This section highlights and details the detailed length of model training.
- Originally used columns for navigation and layout, however this caused the information to be difficult to read so a radio option was added instead, this allowed for easier and clearer navigation. 
- It includes the best results from keras tuner during hyperparameter optimisation, the best hyperparameters were used as the main values for the final model.
- Plots of training and validation accuracy for: 
1. The best_model (tuner.search() model).
2.  The cross validation model (using best_model hyperparameters).
3. The final model, trained using a full dataset (combination of train and validation sets).
- The cross validation model also provides metrics confirming generalisation performance over multiple folds.
- Classification Report and a Confusion Matrix for the best_model, cross_val_model and final_model.


### Powdery Mildew Detector

This section contains a link to a live dataset for the client to download along with the final model.
- Image uploader: allows client to upload an image and instantly get the classification prediction along with a prediction probability.
- If the prediction probability it greater than 0.5 the image will be classified as powdery mildew else it will be classified as healthy.

## Bugs

### Reproducibility:

One difficulty during the building and training of the model was reproducibility. After working on the project for a day and coming back and having to rerun all the cell blocks the output of fitting and training was always different meaning the analysis of history plots or confusion matrix's had to be updated. This became quite frustrating as I already set random_state to a specific number, 25 when building the datasets. However this wasn't enough, I had to create seed=SEED where SEED=25 a number of times to try and keep the notebook as reproducible as possible. I set the os.environ['PYTHONHASHSEED'] = str(SEED) and os.environ['TF_DETERMINISTIC_OPS'] = '1' and set random.seed, np.random.seed, tf.random.set_seed all to SEED. My conclusion is there will always be an issue with trying to reproduce model performance when fitting or evaluating the test set. I believe I tried to control the outcome as much as possible and even though the result might be different, one evaluation would not be too dissimilar from another evaluation when running the same evaluation on different days.

### Slug Size:

Slug Size did become an issue, it might be due to the size of my model as it is a CNN model with 839,105 parameters, however after i downloaded it, it was only 10MBs in size. It is still a large file however the slugsize was over 500MB which caused difficult issues when uploading to heroku. To combat this I added the inputs folder (containing the kaggle dataset) to my gitignore and slugignore files. However this wasn't enough, I ended up having to add my final model to the gitignore and slugignore file as well and then load it in from a separate source. I first tried google drive, however it worked when running the streamlit app from the terminal, however when trying to deploy on heroku it couldn't download the model. I then opted for using a [hugging face space](https://huggingface.co/) as it works better when trying to deploy with heroku. This proved successful and I was able to successfully deploy my app to heroku and have the final model available. 


## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

### [Numpy](https://jupyter.org/):
- Used to handle image arrays and numerical operations, normalisation.
- Example: np.array(img) / 255.0
### [Pandas](https://pandas.pydata.org/):
- This was used to manage and manipulate datasets, such as image paths, labels, CSV logs. 
- Example: pd.read_csv('outputs/logs/final_model_history.csv')
### [Matplotlib](https://matplotlib.org/):
- Matplotlib was used for image plotting during image analysis, along with performance metrics, training history plots.
- Example: plt.imshow(avg_image) 
### [Seaborn](https://seaborn.pydata.org/):
- This was used to create advanced plot during label distribution visualisation.
- Example: sns.countplot(x=y_labels)
### [Tensorflow](https://www.tensorflow.org/):
- This was used to build, tune, train and evaluate my convolutional neural network (CNN) model.
- Example: model.fit(), load_model()
### [Scikit Learn](https://scikit-learn.org/stable/):
- This was used for classification reports, cross validation, confusion matrixs.
- Example: from sklearn.model_selection import StratifiedKFold, skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
- Example: Also used for train_test_split during dataset creation 
### [PIL](https://pypi.org/project/pillow/):
- This was used to load and preprocess images.
- Example: Image.open(image_path).convert('RGB')
### [Keras](https://keras.io/):
- Used for model tuning and label encoding, building models.
- Example: y_train_encoded = label_encoder.fit_transform(y_train)
- Note: keras is part of tensorflow however I had to import it as keras.models or keras import layers

## Credits

Here is a list of documents I used to help create my project.

1. [Jupyter Notebook Documentation](https://docs.jupyter.org/en/latest/) 
2. [Python Documentation](https://docs.python.org/) 
3. [Streamlit Documentation](https://docs.streamlit.io/) 
4. [Stack Overflow](https://stackoverflow.com/)
5. [Keras Documentation](https://keras.io/api/)
6. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
7. [Numpy Documentation](https://numpy.org/doc/)
8. [Slack](https://app.slack.com/)
9. [Pandas](https://pandas.pydata.org/docs/)
10. [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
11. [Seaborn Documentation](https://seaborn.pydata.org/)
12. [Kaggle](https://www.kaggle.com/)
13. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
14. [Code Institute](https://codeinstitute.net/)
15. [PIL Documentation](https://pillow.readthedocs.io/en/stable/)

- The Code Institute modules were very useful in helping understand workflow and structure of my jupyter notebook. 
- The Code Institute [mildew detection template](https://learn.codeinstitute.net/courses/course-v1:CodeInstitute+PA_PAGPPF+2/courseware/bde016cdbd184cdeafd341a73807e138/bd2104eb84de4e48a9df6f685773cbf2/) was a very helpful base for the readme file, I also use the recommended business case.
- The Code Institute Walkthrough [malaria detector](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+3/courseware/07a3964f7a72407ea3e073542a2955bd/29ae4b4c67ed45a8a97bb9f4dcfa714b/) project also gave insight for structure and general code practice of jupyter notebooks.