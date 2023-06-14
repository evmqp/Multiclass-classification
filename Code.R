wine <- read.csv("winequality-red.csv")


# Define a function to normalize a vector using min-max scaling
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply the function to each column of the dataset, except the quality column
wine_normalized <- as.data.frame(lapply(wine[-12], normalize))

# Add the quality column back to the normalized dataset
wine_normalized$quality <- wine$quality

# Print the first few rows of the normalized dataset
head(wine_normalized)

wine <- wine_normalized

# Explore the dataset
str(wine)
summary(wine)

# Calculate the frequency of each quality value
quality_count <- table(wine$quality)

# Plot a bar chart using the barplot function
barplot(quality_count,
        xlab = "Quality", # Label for the x-axis
        ylab = "Count", # Label for the y-axis
        col = "lightblue", # Color for the bars
        main = "Bar Chart of Wine Quality") # Title for the plot

# Convert the quality variable into a factor with 3 levels: low, medium, high
wine$quality <- cut(wine$quality, breaks = c(0, 5, 7, 10), labels = c("low", "medium", "high"))

# Split the dataset into training and testing sets (70:30 ratio)
set.seed(123)
train_index <- sample(nrow(wine), 0.67 * nrow(wine))
train <- wine[train_index, ]
test <- wine[-train_index, ]

# Define a function to calculate the softmax function
softmax <- function(x) {
  # Subtract the maximum value of each row from x
  x <- x - apply(x, 1, max)
  # Apply the exponential function and divide by the row sums
  exp(x) / rowSums(exp(x))
}

# Define a function to perform logistic regression with gradient descent
logistic_regression <- function(x, y, alpha, iter) {
  # x: matrix of predictors (with intercept column)
  # y: matrix of response (one-hot encoded)
  # alpha: learning rate
  # iter: number of iterations
  
  # Initialize the parameters randomly
  theta <- matrix(runif(ncol(x) * ncol(y), -1, 1), ncol = ncol(y))
  
  # Initialize a vector to store the cost values
  cost_vec <- numeric(iter)
  
  # Loop over the iterations
  for (i in 1:iter) {
    # Calculate the predicted probabilities using softmax function
    p <- softmax(x %*% theta)
    
    # Calculate the gradient of the cost function
    grad <- t(x) %*% (p - y)
    
    # Update the parameters using gradient descent
    theta <- theta - alpha * grad
    
    p <- p + 1e-10
    cost <- -sum(y * log(p))
    cost_vec[i] <- cost
    
    # Print the cost function every 100 iterations
    if (i %% 100 == 0) {
      
      cat("Iteration:", i, "Cost:", cost, "\n")
    }
  }
  
  data_cost <- data.frame(iteration = 1:iter, cost = cost_vec)
  
  # Return the final parameters
  return(list(theta=theta, data_cost=data_cost))
}

# Prepare the predictors and response for training and testing sets
x_train <- as.matrix(cbind(1, train[, -ncol(train)])) # add intercept column
y_train <- model.matrix(~ quality - 1, data = train) # one-hot encode the response
x_test <- as.matrix(cbind(1, test[, -ncol(test)])) # add intercept column
y_test <- model.matrix(~ quality - 1, data = test) # one-hot encode the response

# Train the logistic regression model using gradient descent
result <- logistic_regression(x_train, y_train, alpha = 0.01, iter = 1000)

theta <- result$theta

library(ggplot2)

ggplot(result$data_cost, aes(x = iteration, y = cost)) +
  geom_smooth(method = "loess", color = "red") +
  theme_bw() +
 labs(title = "Cost function vs iteration", x = "Iteration", y = "Cost")


# Predict the probabilities for the testing set
p_test <- softmax(x_test %*% theta)

# Convert the probabilities into class labels
y_pred <- levels(train$quality)[apply(p_test, 1, which.max)]

# Calculate the accuracy of the predictions
accuracy <- mean(y_pred == test$quality)
cat("Accuracy:", accuracy)