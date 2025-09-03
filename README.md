# Machine Learning Digit Classification

## Overview
This assignment implements machine learning models including perceptron, neural networks, and recurrent neural networks for various tasks including digit classification and language identification.

## Project Structure

### Files You'll Edit:
- `models.py` - Main implementation file containing all model classes

### Provided Files:
- `nn.py` - Neural network mini-library with node operations
- `backend.py` - Backend code for datasets and visualization
- `autograder.py` - Assignment autograder for testing
- `data/` - Datasets for digit classification and language identification

## Installation Requirements

Install required dependencies using conda:
```bash
conda activate [your environment name]
pip install numpy matplotlib
```

Test your installation:
```bash
python autograder.py --check-dependencies
```

## Assignment Tasks

### Q1: Perceptron (6 points)
Implement a binary perceptron classifier that separates data points into classes (+1 or -1).

**Implementation Requirements:**
- `run(self, x)` - Compute dot product of weights and input
- `get_prediction(self, x)` - Return 1 if dot product ≥ 0, else -1
- `train(self, dataset)` - Train until 100% accuracy on training data

**Test:** `python autograder.py -q q1`

### Q2: Non-linear Regression (6 points)
Train a neural network to approximate sin(x) over [-2π, 2π].

**Implementation Requirements:**
- `__init__()` - Initialize network parameters
- `run(x)` - Forward pass returning predictions
- `get_loss(x, y)` - Compute square loss
- `train(dataset)` - Train using gradient descent

**Success Criteria:** Loss ≤ 0.02 on training data

**Suggested Architecture:**
- Hidden layer size: 512
- Batch size: 200
- Learning rate: 0.05
- One hidden layer

**Test:** `python autograder.py -q q2`

### Q3: Digit Classification (6 points)
Classify handwritten digits from MNIST dataset (28x28 pixel images).

**Implementation Requirements:**
- Network input: 784-dimensional vectors (flattened 28x28 images)
- Network output: 10-dimensional scores (one per digit class 0-9)
- Use `nn.SoftmaxLoss` for classification
- No ReLU in final layer

**Success Criteria:** ≥97% accuracy on test set

**Suggested Architecture:**
- Hidden layer size: 200
- Batch size: 100
- Learning rate: 0.5
- One hidden layer

**Test:** `python autograder.py -q q3`

## Neural Network Framework (`nn.py`)

### Key Node Types:
- `nn.Parameter(n, m)` - Trainable parameters (n×m matrix)
- `nn.Constant` - Input features/labels (provided by dataset)
- `nn.Linear(features, weights)` - Matrix multiplication
- `nn.AddBias(features, bias)` - Add bias vector
- `nn.ReLU(x)` - Rectified Linear Unit: max(x, 0)
- `nn.Add(x, y)` - Element-wise addition

### Loss Functions:
- `nn.SquareLoss(a, b)` - For regression tasks
- `nn.SoftmaxLoss(logits, labels)` - For classification tasks

### Training Utilities:
- `nn.gradients(loss, [params])` - Compute gradients
- `nn.as_scalar(node)` - Extract Python number from node
- `parameter.update(direction, multiplier)` - Update parameters

## Dataset Usage

Iterate through batches:
```python
for x, y in dataset.iterate_once(batch_size):
    # Training step
    pass
```

Check validation accuracy:
```python
accuracy = dataset.get_validation_accuracy()
```

## Architecture Design Tips

1. **Start Simple:** Begin with one hidden layer, then add complexity
2. **Learning Rate:** Critical hyperparameter - too high causes divergence, too low causes slow learning
3. **Batch Size:** Smaller batches need lower learning rates
4. **Layer Sizes:** 10-400 neurons typically work well
5. **Debugging:** If getting NaN/Infinity, reduce learning rate

## Recommended Hyperparameter Ranges:
- Hidden layer sizes: 10-400
- Batch size: 1 to dataset size
- Learning rate: 0.001-1.0
- Hidden layers: 1-3

## Testing
Run individual questions:
```bash
python autograder.py -q q1  # Perceptron
python autograder.py -q q2  # Regression
python autograder.py -q q3  # Classification
```

Run all tests:
```bash
python autograder.py
```

## Submission
Submit `models.py` in a zip file named `<your_UIN>.zip` to Canvas.

## Notes
- Autograder should complete within 20 seconds for correct implementations
- Due to randomness in initialization, occasional failures may occur
- Staff solutions take 2-12 minutes total runtime
- Keep track of architectures and hyperparameters you try
