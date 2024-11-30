## Neural Network from Scratch

1. **Softmax Regression**: Implementation of the softmax loss function and its gradient computation.
2. **Stochastic Gradient Descent (SGD)**: A basic SGD optimizer, with an extension for momentum.
3. **Least Squares Optimization**: A demonstration of how SGD optimizes a simple linear regression problem.

## How to Run

1. **Set up environment**:
   ```bash
    make all 
    ```

## Tasks 

1.1 - **Gradient test and softmax loss implementation**

```bash
python neural_net_from_scratch/tests/test_gradients.py
```

see plots in:

```bash
neural_net_from_scratch/artifacts/gradient_test.png
```
Softmax implementation in 

```bash
neural_net_from_scratch/src/losses/sofmax_loss.py
```

1.2 -  **SGD Optimization for least square problem**

run 
```bash
python nneural_net_from_scratch/tests/test_sgd.py
```

see :
1. plots in 
```bash
neural_net_from_scratch/artifacts/sgd_vs_analytical.png
```

2. SGD implementation in 
```bash
neural_net_from_scratch/src/losses/least_squares.py
```
