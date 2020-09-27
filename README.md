# Audio-Classification-Models
Audio classification is a popular topic, here I implement several models using TenserFlow and Keras.
As part of code implementation of article https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820#80af and paper 
[Environment Sound Classification using Multiple Feature Channels and Attention based Deep Convolutional Neural Network].

### 1. Simplest Audio Features based Classification
Some traditional audio features like zero-crossing rate, averaged MFCC, RMS, averaged STFT/Mel Spectrogram are used, followed by MLP(if concatenated and padded) or 2DCNN(if concatenated horizontally). It's a simple baseline of audio classification tasks.
Because audio files have diferent durations, a slidding window is employed to get same-length sub sequence.
##### The codes in Part 1 can be found in audiofeature_2dcnn.py and audiofeature_mlp.py
### 2. Deep Neural Network Models
#### 2.1 Recurrent baseline

Directly using Time-Frequency Spectrogram(STFTs, Mel Spec) as input + LSTM. For this purpose, we (in our code) treat the spectrogram as a time-series of dimension 128, and feed this into a single lstm layer with 256 cells. The output of the LSTM layer is then fed into a fully-connected layer with 64 neurons, which feeds into our output layer which is a softmax with n output nodes corresponding to each of our n classes. The layers used have been depicted in the figure below:

![Image text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/direct_LSTM.jpg)

Similarly, we can use moving window/resize to get inputs with same shape, for this model, the spec matrix is reansposed to make sure it's time frequency first.

![Image text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/spec_feature.png)
#### 2.2 Conv-LSTM-DNN(CLDNN) Model

The Convolutional part consists of 3 convolutional layers and 2 max pool layers. The convolutional layers have rectangular filters of size [1, 3] and [3, 1] such that they convolve in frequency and time separately ( For more specific detail please refer to section 2.6). Also, the stride of the max pool layer is along the frequency and time dimension separately. The output of the convolutional segment has shape [32x32x64], which is reshaped to [32, 2048] so as to retain the sequence structure of our input feature. This output is then fed into a stacked-LSTM layer consisting of two stacked LSTMS with hidden dimension 128. The output from the last time-step of the stacked LSTM is then fed into two fully connected layers consisting of 64 and n(no. of classes) nodes respectively. The architecture has been depicted below (This architecture is the original version in article, I modified it in the code):

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/CLDNN.jpg)

All the inputs have identical shape [128x128] by moving window with window length 128 and overlap length 64. One sample is like this:

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/spec_feature_notranspose.png)

Note: the original input is frequency dimension × time dimension, after CNN block the feature has to be transposed before feeding into RNN block.
#### 2.3 Joint Auto-Encoder with supervised Classifier
we attempt to tackle the problem as a multi-task problem, where we jointly train an auto-encoder and a classifier. The encoder segment is shared by both these tasks, and consists of 4 conv+max pool layers, which bring down the input dimension to [8, 8] with 128 filters. This is then fed into a dense layer of 64 units, which acts as our latent descriptor. The decoder then reverses the operation performed by the encoder and computes the reconstruction loss as the mean square error for each pixel of the reconstructed and original images. The genre classifier is constructed using 3 fully connected layers at the output of the latent descriptor, which is designed to minimize the loss in classification. The architecture has been envisioned below:

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/JAE.jpg)

The input of this model is the same as the previous model.
#### 2.4 SoundNet based Convolutional Architecture

The SoundNet Architecture typically consists of two conv+max pool layers followed by 3conv layers, a max pool layer and finally 3 more conv layers. The convolutional blocks have increasingly higher number of filters and inversely, decreasing filter size as the depth of the model increases. For our model, we simulate most of the architecture with minor changes in filter size. The first two convolutional layers have stride 8 each so as to bring the input dimension down to a reasonable scale. We also reduce the final convolutional block to 2-conv layers. The output of this so called SoundNet block (of dimension 6x1024) is then flattened fed into three densely connected layers with 512, 128 and finally n nodes respectively.

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/SoundNet_CNN.jpg)

The original audio sequence is splitted into several equal-length sub sequence.

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/sub_wav.png)
#### 2.5 SoundNet based Recurrent Architecture

We remove the final convolutional-layer, which changes our output dimension of the SoundNet block to 13x256. We treat this as a time-sequence of 13 time-steps with dimension 256 and feed this to a 2-layer stacked-LSTM of 256 cells each. The rest of the model follows from previous architecture.

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/SoundNet_LSTM.jpg)
#### 2.6 Single Feature Channel and Deep Convolutional Neural Networks
In this method, a 2d spectrogram (generated either by STFT or Mel/Gammatone) is fed into a CNN model directly. Each feature set is of the shape (f, t, c), where t is the compressed time domain (compressed due to window size and hop length) and c is the number of channels. Each window of time yields f number of features (f=128 in our model). So, we treat the time domain and the feature domain separately. The kernels with the form [1×m] work on the feature domain and the ones with [n×1] work on the time domain. Using the [1×m] type of convolution operation enables the network to process each set of features from a time window separately. And, the [n×1] type of convolution allows the aggregation of a feature along the time domain. So, each kernel works on each channel, which means that all different types of features extracted from the signal feature extraction techniques is aggregated by every kernel. Each kernel can extract different information from an aggregated combination of different feature sets. Another major advantage of using these type of convolutions is the reduction in number of parameters. For a kernel of size [1×m], one dimension of the kernel is 1, it has the same number of parameters as a 1D convolution of kernel size m. But, it has the operational advantage of 2D convolution, as it works on two spatial dimensions.If the input is square, m and n could have identical value.

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/Multi-channel-Sparable-CNN.png)

##### All the codes in Part 2 could be found in single_channel_models.py and soundnet_models.py

### 3. Multiple Feature Channels and Deep Convolutional Neural Networks
In this method, several 2d spectrograms (generated by Mel-Frequency Cepstral Coefficients (MFCC)/Gammatone Frequency Cepstral Coefficients (GFCC)/the Constant Q-transform (CQT)/Chromagram) is fed into a multi-channel CNN model. The rest of the model follows from previous architecture in section 2.6. Some of the features you can use are shown below:

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/Multi-channel.png)

Another methodology is that we treat STFTs spectrogram separately as real/imaginary part, or to say magnitude/phase part instead of taking the absolute of complex values. It is because they contain different information. For each frequency bin, the magnitude ![](http://latex.codecogs.com/gif.latex?\\sqrt{im^2+re^2}) tells you the amplitude of the component at the corresponding frequency. The phase atan2(im, re) tells you the relative phase of that component. So here we combine these two parts into a 2-channel feature and feed into our network.
##### All the codes in Part 3 could be found in multi_channel_models.py

### 4. Sub-Spectral Net
According to this paper: "Subspectralnet–using sub-spectrogram based convolutional neural networks for acoustic scene classification", which assumes that the key patterns of each class are concentrated on the specific frequency bins, and those key patterns are effectively learned by dividing the input mel-spectrogram into the multiple sub-bands and using the CNN classifier for each sub-band independently. By dividing the input mel-spectrogram into the multiple sub-bands and learning the CNN classifier of each sub-band independently, the key patterns of specific frequency bins are effectively learned. The CNN classifiers consist of 4 convolution blocks and 1 fully connected network as in the network of section2.6. The outputs of all FCN layers are concatenated, and they are used as the input of the global classifier.

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/SubSpec_Net.png)
##### All the codes in Part 4 could be found in subspec_net.py

### 5. Other tricks
#### 5.1 Noise Reduction and Silence Removement
What if the input data are noisy? Here we use a denoise methodology called "Spectral gating", for details of this algorithem please refer to https://timsainburg.com/noise-reduction-python.html. When you are using this method, you have to determine what your noise part is (parameter "noise_clip"), after experiment, just pass the entire audio in as the noise part will get the best result if it's hard to locate the noice clip. Here is a comparison between with/without noise reduction in an audio:

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/denoise_before.png) ![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/denoise_after.png)

Besides, if you want to remove the silence parts in your data, the "trim" and "split" function in Librosa package will help a lot. Here is an exapmle of the "split" function；

![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/split_before.png) ![Image_text](https://github.com/WWH98932/Audio-Classification-Models/blob/master/images/split_after.png)
#### 5.2 SOTA Image Classification Models and Recurrent Models
According to our experience, CRNN model performs very well in audio-classification tasks. There are already lots of elegant CNN architectures such as ResNet, so we can simply modify them and combine them with RNN models. I gave an example of ResNet+LSTM here, you can try it with other models like VGG, Inception, Resnext, DenseNet... with bidirectional rnn models and see if they perform well.
#### 5.3 Should we employ Attention mechanism？
If your data is human voise, or some other sound which has strong local prensentation, it's naturally that Attention could be used here. Indeed, there are also a lot of papers are doing so, both Attention and Self Attention. I put a simple example of Attention and Self Attention in the code, but the tricky part is, where to put this layer at. In our experiment, when applying different layers, including CNN layers and RNN layers. The experimental results showed that applying attention for RNN layers obtained highest accuracy. However, we found when applying attention for CNN layers, the performance is not always improved. I'm going to explore this in our future work. 
