# Music Genre Classification


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
      
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#notebooks">Notebooks</a>
      <ul>
        <li><a href="#feature-extraction">Feature Extraction</a></li>
        <li><a href="#visualising-mfccs">Visualising MFCCs</a></li>
        <li><a href="#models-using-covariance">Models using MFCC Covariance Matrix</a></li>
        <li><a href="#models-using-spectrogram">Models using Mel Spectrogram (including Convolutional Neural Network)</a></li>
      </ul>
    </li>
      
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to use several models to classify musical genres based on audio samples and different visualisation techniques to understand the data.
 
This project is inspired by the code on https://data-flair.training/blogs/python-project-music-genre-classification/, which implements a K-Nearest Neighbour approach to the problem. That served as a starting point to this project.
 
Dataset: http://marsyas.info/downloads/datasets.html


<!-- Notebooks -->
## Notebooks

### Feature Extraction

Extracted Mel Frequency Cepstral Coefficients (MFCCs) from audio samples. Includes K-Nearest Neighbours approach to classifying genres (from https://data-flair.training/blogs/python-project-music-genre-classification/).

### Visualising MFCCs

Visualised Mel Frequency Cepstal Coefficients using colormaps to better understand the data and gain a more intuitive perspective on MFCCs. Compared MFCCs for different genres.

### Models using MFCC Covariance Matrix

Converted MFCC mean and covariance matrix features into Pandas dataframe. Trained a logistic regression model to classify music genres using these features. Tuned model to prevent overfitting by increasing the strength of regularisation and randomising the data. Explored the impact of using PCA to reduce the number of features. 

### Models using Mel Spectrogram (including Convolutional Neural Network)

Used Librosa to extract Mel spectrogram from audio samples. This unstructured data works better for deep learning models like Convolutional Neural Networks. Cleaned data to prepare it for the CNN. Working on preventing the model from overfitting by:
- Adding data (another notebook coming soon aimed at data augmentation)
- Reducing complexity of CNN
- Applying dropout

