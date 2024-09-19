# Supervised-Machine-learning-regression-project-1
Supervised machine learning project linear regression single variable and multi variable for ecommerce customers data to predict yearly amount spent for customers
# Linear Regression Training Project: Ecommerce Clients

## What is the objective of the project?
This is a training project. This means that the data is not real and the project is for education purposes only. We suppose that a company is trying to decide whether to focus their efforts on their mobile app experience or their website. We are here to help them make a data-driven decision.

## What is the data in the project ?
In this project we work with a dataset [available on Kaggle](https://www.kaggle.com/iyadavvaibhav/ecommerce-customer-device-usage). The data includes information about customers of an e-commerce website, including the following:
- Avg. Session Length: Average session of in-store style advice sessions.
- Time on App: Average time spent on App in minutes
- Time on Website: Average time spent on Website in minutes
- Length of Membership: How many years the customer has been a member.
Note that all the personal information is not real.

Here’s a sample README file for your linear regression project:

---

# Linear Regression Model

This project implements a simple linear regression model from scratch using Python. The model is trained to predict values based on a single feature and has been evaluated for accuracy using the R² score.

## Project Overview

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables.

## Key Components

1. **Cost Function**:
   The cost function is used to calculate the error (mean squared error) between the predicted and actual values. It ensures the model optimizes its parameters (weight and bias) by minimizing this error.

2. **Gradient Descent**:
   The gradient descent algorithm is employed to optimize the parameters (weight `w` and bias `b`) by iteratively updating them to reduce the cost. We calculate the partial derivatives with respect to `w` and `b` and update them using a learning rate (`alpha`).

3. **Model Training**:
   The model is trained using a loop over a set number of iterations, where the cost is calculated, and the gradients are used to adjust the parameters.

4. **Scaling**:
   Feature scaling is applied to both the input feature `x` and the output target `y`. This helps the gradient descent algorithm converge faster.

5. **Accuracy Evaluation**:
   The model's performance is evaluated using the R² score, which measures how well the model fits the data. The resulting model has an accuracy of **65.5% For single linear regression and 85% for multiple linear regression**.

## Code Structure

- **`cost_func(x, y, w, b)`**:
  - Computes the cost (mean squared error) for the given parameters `w` and `b`.

- **`gradient(x, y, w, b)`**:
  - Calculates the gradients for `w` and `b` using the training data.

- **`gradient_descent(x, y, w, b, alpha, num_iter)`**:
  - Implements the gradient descent algorithm to update the parameters over multiple iterations.

- **`predict(x, w, b)`**:
  - Predicts the target values for given input `x` using the learned parameters.

- **`calculate_r_squared(x, y, w, b)`**:
  - Calculates the R² score to evaluate the model's accuracy.

## How to Run

1. Clone or download this repository.
2. Install required dependencies (if any), such as `pandas`, `numpy`, `matplotlib`, and `seaborn`.
3. Load your dataset or use the example dataset provided.
4. Run the script to train the model and view the accuracy.

```bash
python linear_regression.py
```

## Results

The model achieves an accuracy (R² score) of **65.5% For single linear regression and 85% for multiple linear regression**. This can be improved by adjusting the learning rate, increasing iterations, or using more sophisticated techniques like multiple-variable regression.

---

You can add more details based on the dataset and specific usage instructions. Let me know if you'd like any modifications!
