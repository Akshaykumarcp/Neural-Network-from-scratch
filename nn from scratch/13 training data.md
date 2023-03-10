### Why data is scaled 0 to 1 or -1 to 1

- scaling is ideal is a neural network’s reliance on many multiplication
operations.
- If we multiply by numbers above 1 or below -1, the resulting value is larger in scale
than the original one.
- Within the -1 to 1 range, the result becomes a fraction, a smaller value.
- Multiplying big numbers from our training data with weights might cause floating-point overflow
or instability — weights growing too fast.
- It’s easier to control the training process with smaller
    numbers.

### How many samples do we need to train the model?
- Usually, a few thousand per class will be necessary, and a few tens of thousands should be
preferable to start.
- The difference depends on the data complexity and model size.
- If the model has
to predict sensor data with 2 simple classes,
- for example, if an image contains a dark area or does
not, hundreds of samples per class might be enough.
- To train on data with many features and
several classes, tens of thousands of samples are what you should start with.
- If you’re attempting
to train a chatbot the intricacies of written language, then you’re going to likely want at least
millions of samples.
Supplementary