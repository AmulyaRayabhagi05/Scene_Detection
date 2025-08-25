# Scene_Detection
Abstract—The purpose of our project was to apply a Convolutional Neural Network (CNN) to the intel image classifier dataset for scene recognition. Currently there is a trend in using advanced libraries to seamlessly create CNN’s however we were given the challenge of building our very own model using just the basic functions. Our model is built from scratch with the use of NumPy and PyTorch libraries for data processing, loading, and cleaning. 
<br>
<br>
The dataset we used contained over 25,000 RGB images out of which we chose 10,000 to work with. Each image was already cleaned to be 150 x 150 pixels, however we had to downscale it to 64 x 64 pixels for usability in google colab (limited RAM). The images spanned from 6 different classes including buildings, forest, glacier, mountain, sea, and street, however our primary focus was on buildings, forest, glacier and mountain.
<br>
<br>
Our CNN was built using a multitude of techniques. We started with a convolutional layer using eight 3×3 filters for feature extraction. Then we used a ReLU activation for non-linearity. We also used a 2×2 max pooling layer to reduce spatial dimensions. Followed by that we had a flattening stage and finally a fully connected output layer with a Softmax activation for multi-class classification. We were able to use backpropagation in order to have iterative weight changes and our model was trained on stochastic gradient descent (SGD) and cross-entropy loss.
<br>
<br>
In order to get the best hyperparameters for our model we changed a variety of conditions. Some of the most varying objectives that we changed included: varying learning rates, epochs, and dataset sizes. The best model that we were able to create had a learning rate of 0.0043 over 60 epochs and was able to achieve an accuracy of 63.90%. We then were able to create metrics to understand the results that we had obtained by creating a confusion matrix, precision recall matrix and even ROC curves. These metrics show that four out of the six categories were able to train much better than those that were not in the reduced version of the training dataset.
<br>
<br>
Overall we were able to successfully build a manual CNN model without using any prebuilt libraries. The results that we were able to obtain show that even with constraints such as the usage of RAM we were able to get some optimal results. Some of our future recommendations for this project include, getting to train on higher quality images like the original 128 x 128 size by using a GPU instead of google colab. We could also try using data augmentation using torchvision.transforms during our preprocessing that would allow us to do random flips with our images. Finally the model that we created is considered to be a really shallow model as we looked at mostly shapes and edges. By increasing the number of layers and creating a more indepth model we could expand into higher level layers including textures, objects and scenes.
<br>
<br>
Keywords—convolutional neural network, NumPy, PyTorch, image classifier
