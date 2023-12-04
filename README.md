# Melanoma Detection Assignment
> Problem statement: The objective is to construct a Convolutional Neural Network (CNN)-based model capable of precise melanoma detection. Melanoma, a potentially lethal form of cancer, contributes to 75% of skin cancer-related fatalities. Detecting melanoma early is critical for effective intervention. Developing a solution that can analyze images and promptly notify dermatologists about the existence of melanoma has the potential to significantly decrease the manual effort required in the diagnostic process.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Our goal is to create a Convolutional Neural Network (CNN)-powered model that can precisely recognize instances of melanoma. Melanoma, a type of skin cancer, presents a serious risk when not identified early, contributing to a substantial percentage of fatalities related to skin cancer (approximately 75%). The development of a solution adapt at analyzing images and swiftly alerting dermatologists to the existence of melanoma has the potential to alleviate the significant manual workload currently associated with the diagnostic process.

- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following nine diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used
- Python: The primary programming language employed for coding throughout the project.
- Git and GitHub: Employed for version control and collaborative development of code.
- Git Bash: A command-line interface facilitating Git operations on Windows.
- TensorFlow: An open-source machine learning framework applied for constructing and training neural network models.
- Matplotlib: A versatile plotting library used for generating visualizations such as training curves and images.
- NumPy: A powerful library for numerical computations in Python, utilized for array manipulation and mathematical operations.
- PIL (Pillow): Python Imaging Library, utilized for loading and manipulating images.
- Augmentor: A library dedicated to data augmentation, employed for the generation of augmented images.
- Seaborn: A statistical data visualization library built on Matplotlib, employed for creating informative plots.
- Google Colab: An online platform facilitating the creation and execution of Python code, commonly used for data analysis and machine learning.


## Conclusions
1. **Model 1:Minimal Dropout strategy** 
    - The model was trained for ~ 20 epochs.
    - The training loss decreased progressively over the epochs, indicating the model's learning and convergence.
    - Training accuracy improved with each epoch, reaching ~80%.
    - However, there's a noticeable discrepancy when it comes to the validation dataset, where the accuracy is significantly lower at ~ 55% which indicates a potential       **overfitting** issue in the model. 

2. **Model 2:Model with additional Dropouts** 
   - The model was trained for ~ 20 epochs.
   - The close similarity between training accuracy and validation accuracy, both at a low level, suggests that the model is experiencing **underfitting**. 
   - Training loss is minimal, the validation loss exhibits fluctuations, indicating potential challenges in generalizing to new data.
   
3. **Model 3:Model with Data Augmentation approach** 
   - The model was trained for ~ 20 epochs.
   - Training and Validation accuracy was in between close to 45 to 48% which indicates overfitting is resolved to an extent.
   - Model performance enhanced through the use of Data Augmentation.
   - The convergence of training accuracy and validation accuracy suggests a good fit, yet the overall accuracy remains relatively low.
   - To address this issue, additional epochs are needed, taking into account the handling of **class imbalances** in the training data.

4.  **Model 4:Modelling Augmented data (after class imbalance)** 
     - The model was trained for ~ 30 epochs.
     - Training and Validation accuracy were approximately 90 % and 80% respectively.
     - Data augmentation helped improve model performance by generating additional diverse samples for each class.
     - Class rebalance has mitigated overfitting to some extent.   
     - Can be considered as **Final model** than compared to previous 3 models, however it can be enhanced / fine-tuned further.
   
Overall, the trained CNN model demonstrates promising results in accurately classifying skin cancer images, with effective handling of class imbalances and avoidance of overfitting.Leveraging Google Colab streamlined the development process, facilitating efficient and effective experimentation.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python: The programming language used for writing the entire code.
- Git and GitHub: Used for version control and collaborating on code.
- Git Bash: A command-line interface for Git on Windows.
- GitHub: A platform for hosting and collaborating on Git repositories.
- TensorFlow: An open-source machine learning framework used for building and training neural network models.
- Matplotlib: A plotting library used for creating visualizations, such as training curves and images.
- NumPy: A library for numerical computations in Python, used for working with arrays and mathematical operations.
- PIL (Pillow): Python Imaging Library, used for loading and manipulating images.
- Augmentor: A library for data augmentation, used for generating augmented images.
- Seaborn: A statistical data visualization library based on Matplotlib, used for creating bar plots.
- Google Colab: An online platform for writing and executing Python code, often used for data analysis and machine learning.

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- The motivation behind this project stemmed from the imperative for proficient melanoma detection.
- The training and evaluation phases relied on the "Skin cancer ISIC The International Skin Imaging Collaboration" dataset.
- The project synthesized insights from diverse sources to formulate a potent CNN model.

## Contact
Created by [@mabusaleha] - feel free to contact me!