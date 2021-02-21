# Music Genre Classification

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to use several models to classify musical genres based on audio samples and different visualisation techniques to understand the data.
 
This project is inspired by the code on https://data-flair.training/blogs/python-project-music-genre-classification/, which implements a K-Nearest Neighbour approach to the problem. That served as a starting point to this project.
 
Dataset: http://marsyas.info/downloads/datasets.html


<!-- Notebooks -->
## Notebooks

### <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/1-Feature-Extraction.ipynb">1 - Feature Extraction</a>

Extracted Mel Frequency Cepstral Coefficients (MFCCs) from audio samples. Includes K-Nearest Neighbours approach to classifying genres (from https://data-flair.training/blogs/python-project-music-genre-classification/). Compared accuracy of models with different K values.

### <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/2-Visualising%20MFCCs.ipynb">2 - Visualising MFCCs</a>

Visualised Mel Frequency Cepstal Coefficients using colormaps to better understand the data and gain a more intuitive perspective on MFCCs. Compared MFCCs for different genres.

### <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/3-Models%20using%20Covariance%20matrix.ipynb">3 - Models using MFCC Covariance Matrix</a>

Converted MFCC mean and covariance matrix features into Pandas dataframe. Trained a logistic regression model to classify music genres using these features. Tuned model to prevent overfitting by increasing the strength of regularisation and randomising the data. Explored the impact of using PCA to reduce the number of features. 

### <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/4-Models%20using%20Mel%20Spectogram%20(including%20CNN).ipynb">4 - Models using Mel Spectrogram (including Convolutional Neural Network)</a>

Used Librosa to extract Mel spectrogram from audio samples. This unstructured data works better for deep learning models like Convolutional Neural Networks. CNN model has 90% accuracy on test data. At the end of the notebook, one can listen to the audio sample and then see what the actual genre of the music is, as well as a barplot of the model's scores for each genre for the sample.

### <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/5%20-%20Predicting%20music%20genre%20live.ipynb">5 - Predicting music genre live</a>

Uses microphone input to take 3 second audio samples and classify them with the CNN model to predict the music genre. A matplotlib plot shows the scores the model gives for each of the genres based on the classification of the 3 second sample. This plot is updated in real-time as classifications are made (every 3 seconds). The code can also be run through the python file: <a href="https://github.com/alexpondaven/Music-Genre-Classification/blob/main/genre_classify_mic.py">genre_classify_mic.py</a>

