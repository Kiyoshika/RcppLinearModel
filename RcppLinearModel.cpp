// [[Rcpp::plugins("cpp11")]]

#include <Rcpp.h>
#include <stdio.h>
#include <ctype.h>
#include <algorithm>

using namespace Rcpp;

// Convert a string to lowercase
// @param input_str A string to convert to lowercase
// @return Returns the lower case version of the passed string
std::string to_lower(std::string input_str)
{
  std::transform(
    input_str.begin(), 
    input_str.end(), 
    input_str.begin(),
    [](unsigned char c) { return std::tolower(c); }
    );
  
  return input_str;
}

// taking a numerical derivative of the loss function to be completely generalized
// @param loss_function A function pointer representing the loss function that we're trying to optimize
// @param yhat The predicted value from our model
// @param y The actual (reference) value
// @param x The value of the independent variable at row i, used when taking partial derivatives with respect to a coefficient
// @return The numerical approximation of the derivative of the loss function at row i
double loss_derivative(
    double (*loss_function)(double, double),
    double yhat,
    double y,
    double x,
    bool intercept = false
  )
{
  double deriv = (loss_function(yhat + 0.001, y) - loss_function(yhat, y)) / 0.001;
  return intercept ? deriv : deriv*x;
}

NumericVector get_row(
    DataFrame const& data,
    R_xlen_t row
)
{
  NumericVector row_data(data.size()); // # columns
  for (R_xlen_t col = 0; col < data.size(); ++col)
  {
    NumericVector temp = data[col];
    row_data[col] = temp[row];
  }
  
  return row_data;
}

// Make a prediction given a vector of coefficients and an input data frame
// @param coefficients A vector of coefficients which can be obtained by fitting a model
// @param x A data frame containing multiple observations of the independent variables
// @param row The row of the data frame to make a prediction on
// @param family The (exponential family) distribution of the residuals
// @return A prediction based on the linear combination of coefficients and independent variables
// [[Rcpp::export]]
double predict_point(
    NumericVector coefficients,
    DataFrame x,
    R_xlen_t row,
    std::string family = "gaussian"
)
{
  
  auto link_function = [=](NumericVector const& coefficients, NumericVector const& x_row_vec)
  {
    // the link function for Gaussian residuals is simply the mean (i.e, the linear combination itself)
    if (family == "guassian")
    {
      double sum = 0.0;
      for (R_xlen_t i = 0; i <= x_row_vec.size(); ++i)
      {
        if (i == 0) { sum += coefficients[0]; }
        else 
        {
          sum += coefficients[i] * x_row_vec[i-1];
        }
      }
      
      return sum; 
    }
  }; 
  
  return link_function(coefficients, get_row(x, row));
  
}

// Make a prediction given a vector of coefficients and an input data frame
// @param coefficients A vector of coefficients which can be obtained by fitting a model
// @param x A data frame containing multiple observations of the independent variables
// @param family The (exponential family) distribution of the residuals
// @return A vector of predictions based on the linear combination of coefficients and independent variables
// [[Rcpp::export]]
NumericVector predict(
    NumericVector coefficients,
    DataFrame x,
    std::string family = "gaussian"
    )
{
  NumericVector predictions(x.nrows());
  for (R_xlen_t i = 0; i < x.nrows(); ++i)
  {
    predictions[i] = predict_point(coefficients, x, i, family);  
  }
  
  return predictions;
}

// A generic gradient descent based optimizer
// @param loss_function A function pointer for the loss function
// @param link_function A function to map the linear combination of coefficients and independent variables to the conditional expectation of the dependent variable
// @param x An R data frame containing the independent variables
// @param y An R data frame containing the dependent variable
// @param family The (exponential family) distribution of the residuals
// @param learning_rate The step size to take when adjusting the coefficients according to the gradient
// @param k Print loss iterations k times
// @return A vector of coefficient estimates with the zeroth index being the intercept
NumericVector model_optimizer(
    double (*loss_function)(double, double),
    DataFrame const& x,
    DataFrame const& y,
    std::string family,
    size_t max_iter,
    double learning_rate,
    size_t k
  )
{
  // initialize all coefficients to zero initially
  NumericVector coefficients(x.size() + 1); // +1 for intercept
  NumericVector y_vec = y[0];
  
  double pred, loss;
  // iterate through the data set max_iter times
  // TODO: implement early stopping with a "tolerance" threshold
  for (size_t iter = 0; iter < max_iter; ++iter)
  {
    // print loss K times
    if (iter > 0 && iter % (max_iter / k) == 0) { Rcout << "Loss at iter #" << iter << ": " << loss << "\n"; }
    loss = 0;
    // iterate through data points and update weights
    for (R_xlen_t idx = 0; idx < x.nrows(); ++idx)
    {
      pred = predict_point(coefficients, x, idx);
      loss += loss_function(pred, y_vec[idx]);
      coefficients[0] -= loss_derivative(loss_function, pred, y_vec[idx], true) * learning_rate;
      // update remaining coefficients
      for (R_xlen_t coeff = 1; coeff <= x.size(); ++coeff)
      {
        coefficients[coeff] -= loss_derivative(loss_function, pred, y_vec[idx], get_row(x, idx)[coeff-1]) * learning_rate;  
      }
    }
  }
  return coefficients;
}

// @param x An R data frame containing the independent variables
// @param y An R data frame containing the dependent variable
// @param family The (exponential family) distribution of the residuals
// @param learning_rate The step size to take when adjusting the coefficients according to the gradient
// @param k Print loss iterations k times
// [[Rcpp::export]]
NumericVector linear_model(
    DataFrame const& x, 
    DataFrame const& y, 
    std::string family = "gaussian",
    size_t max_iter = 10000,
    double learning_rate = 0.0001,
    size_t k = 10
)
{
  if (family == "gaussian")
  {
    // for Gaussian residuals, the MLE estimates are derived from minimizing MSE
    // there is also closed form solution but I'm implementing a gradient descent approach
    // to be completely general
    auto loss_function = [](double y, double yhat)
    {
      return (y - yhat)*(y - yhat);
    };
    
    return model_optimizer(loss_function, x, y, family, max_iter, learning_rate, k);
  }
  
  Rcout << "Please specify a valid distributional family.\n";
  return NULL;
  
}
