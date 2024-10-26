# Tensorflow Notes

### Transfer Learning
- Transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem.
- Transfer learning is usually done for tasks where your dataset has too little data to train a full-scale model from scratch

The most common incarnation of transfer learning in the context of deep learning is the following workflow:

1. Take layers from a previously trained model.
2. Freeze them, so as to avoid destroying any of the information they contain during future training rounds.
3. Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.
4. Train the new layers on your dataset.

### Fine-tuning
- consists of unfreezing the entire model you obtained above (or part of it), and re-training it on the new data with a very low learning rate.
- can potentially achieve meaningful improvements, by incrementally adapting the pretrained features to the new data.


## Keras Trainable API
- layers and models have three weight attributes
    - weights:  list of all weights variables of the layer
    - trainable_weights: list of those that are meant to be updated (via gradient descent) to minimize the loss during training
    - non_trainable_weights: list of those that aren't meant to be trained. Typically they are updated by the model during the forward pass

- [training vs trainable method](https://keras.io/getting_started/faq/#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute)

- In general all weights are *trainable weights*. The only built-in layer with non-trainable weights is the *BatchNormalization* layer: uses non-trainable weights to keep track of the mean and variance of its inputs during training

- Layers & models also feature a boolean attribute trainable **Layer Freezing**
    - Setting *layer.trainable to False* moves all the layer's weights from trainable to non-trainable

- Calling compile() on a model is meant to "freeze" the behavior of that model. This implies that the trainable attribute values at the time the model is compiled should be preserved throughout the lifetime of that model, until compile is called again. Hence, if you change any trainable value, make sure to call compile() again on your model for your changes to be taken into account.



## What is forward pass?
In the context of a Keras TensorFlow model like MobileNetV3, the forward pass refers to the process of propagating input data through the network to generate predictions. Here's an explanation of how the forward pass works and how it compares to other operations involving gradient descent and trainable weights:

### Forward Pass in MobileNetV3

The forward pass in MobileNetV3 involves the following steps:

1. Input processing: The input image is preprocessed and fed into the network.

2. Feature extraction: The input passes through a series of convolutional layers, depthwise separable convolutions, and squeeze-and-excitation blocks. These layers extract hierarchical features from the input[1].

3. Global pooling: The final feature maps are pooled to reduce spatial dimensions.

4. Classification: The pooled features are passed through fully connected layers to produce class probabilities[1].

The forward pass is implemented using the model's `call` method in Keras, which is invoked when you use the model on input data:

```python
model = tf.keras.applications.MobileNetV3Large()
predictions = model(input_images)
```

### Comparison with Gradient Descent and Trainable Weights

1. Purpose:
   - Forward pass: Generates predictions from input data.
   - Gradient descent: Updates model weights to minimize the loss function.

2. Direction of computation:
   - Forward pass: Flows from input to output.
   - Backpropagation (used in gradient descent): Flows from output to input, computing gradients[2].

3. Trainable weights:
   - Forward pass: Uses the current values of trainable weights.
   - Gradient descent: Updates the trainable weights based on computed gradients[2].

4. Timing in training:
   - Forward pass: Occurs first in each training iteration.
   - Gradient descent: Follows the forward pass and loss calculation.

5. Computational complexity:
   - Forward pass: Generally less computationally intensive.
   - Gradient descent: More computationally demanding, especially for deep networks.

6. Activation functions:
   - Forward pass: Applies non-linear activation functions (e.g., ReLU, hard swish in MobileNetV3).
   - Gradient descent: Requires computation of activation function derivatives[2].

7. Use in inference:
   - Forward pass: Used in both training and inference.
   - Gradient descent: Only used during training.

8. Optimization:
   - Forward pass: Can be optimized for inference (e.g., using TensorFlow Lite for MobileNetV3).
   - Gradient descent: Optimization focuses on efficient gradient computation and weight updates.

In summary, the forward pass in MobileNetV3 is the process of generating predictions from input data, while gradient descent and operations involving trainable weights are part of the learning process that optimizes the model's parameters. The forward pass uses the current state of trainable weights, while gradient descent updates these weights to improve the model's performance[5][6].

Citations:
[1] https://keras.io/2.15/api/applications/mobilenet/
[2] https://towardsdatascience.com/neural-networks-backpropagation-by-dr-lihi-gur-arie-27be67d8fdce?gi=260f9f76e488
[3] https://theneuralblog.com/forward-pass-backpropagation-example/
[4] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py
[5] https://stackoverflow.com/questions/36740533/what-are-forward-and-backward-passes-in-neural-networks
[6] https://www.nicolaromano.net/data-thoughts/training-neural-networks/
[7] https://stackoverflow.com/questions/66203158/does-back-propagation-and-gradient-descent-use-the-same-logic
[8] https://towardsdatascience.com/neural-networks-forward-pass-and-backpropagation-be3b75a1cfcc?gi=99b66830c470