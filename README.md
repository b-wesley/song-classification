# song-classification

## Abstract:
In this project, we seek to classify the genres of songs using the GTZAN dataset. Our approach relies on a variety of methods, divided into two groups based on the different facets of the dataset. We performed a variety of statistical methods on the features derived from the audio data provided in the data set !!!!!!EMMETT ELABORATE HERE!!!!!! The dataset also contains audiofiles and their corresponding spectrograms, for which we used two different classes of neural networks to predict the genres of the songs. !!!! RESULTS RESULTS RESULTS RESULTS !!!!!!!!

## Introduction
Audio and music data is very unique in the world of machine learning, as it can be represented in several ways. Any audio can be represented either through a raw audio file (wave, mp3, aif, etc.) or a spectrogram. Music in particular can also be represented via midi, allowing this type of data to take advantage of models designed for NLP. Additionally, other features about the audio can be derived and used to make predictions about the audio without even directly using the audio or spectrogram. 

This particular task of genre classification is not a novel task, but it does present a good forum for exploring how different models working with different representations of the same audio can perform, and it gives us insight as to the various tradeoffs between the different approaches to classifying (or performing other tasks on) audio/music data. 

The approaches we used fall under two broad categories: statistical/feature-based methods, and neural networks for working directly with the audio and spectrogram data. We expect these models to work well, as methods such as decision trees and KNN work well with feature-based data, while neural networks perform best with more complex data like audio and images. 

Some of our biggest limitations for these approaches(particularly the neural nets) are computational resources for training and evaluating the network.


## Setup
The GTZAN dataset contains 10,000 30-second audio files from 10 different genres (1,000 songs per genre), as well as spectrograms generated from these audio files. The datset also contains two csv files with data derived from the audio files, one corresponding to the 30-second audio files, and another corresponding to 3-second snippets of audio taken from the 30-second files (resulting in 100,000 datapoints). 

!!! statistical setup !!!!!

For the neural networks, we tested a feedforward convolutional network with various numbers of convolutional layers, filter counts, and regularization over a range of learning rates and numbers of pochs. For the audio, we used a WaveNet inspired by DeepMind's 2018 paper of the same name due to its efficacy at extracting data from audio files (trying various numbers of convolutional/residual blocks, dilation depths, and learning rates). To train these models, we initially developed on our local machines to setup the models, and once they were capable of successfully training for 3 epochs, we used Google Colab to perform the full hyperparameter search to find regions where a good learning rate and other hyperparameters may lie.  

## Results

For the Neural Nets, we have gotten mixed results. So far, the spectrogram-based models work best, attaining a training loss of LOSS HERE, though overfitting slightly. To account for this, we have created a new model that uses batch norm and more dropotu layers, but we have yet to find hyperparameters for this architecture that converge to an effective solution. The audio-only network, on the other hand does not perform well yet due to difficulties with adjusting the parameters and architecture for this model, as it was originally intended for use in applying transformations to audio rather than classifying it. As a result, while we have gotten these models to start training, they have yet to converge to a satisfactory result. 

## Discussion

Though the neural nets do not perform as desired yet, this is simply an issue of further tuning and re-designing. The spectrogram models already are capable of learning the desired function, we now just need to tweak the models so that they train without overfitting. The audio models, while currently failing to predict correctly, are likely not working well due to the fact that this architecture was not initially intended for classification, so adapting it for this purpose will likely need further tweaking to adjust the model to properly extract features before classification. 