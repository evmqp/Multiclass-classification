data(iris)
iris$Species <- as.factor(iris$Species)

# Функция softmax
softmax <- function(x) {
  exp(x) / sum(exp(x))
}

# Функция кросс-энтропийных потерь
cross_entropy_loss <- function(y_pred, y_true) {
  -sum(y_true * log(y_pred))
}

# Функция градиентного спуска
gradient_descent <- function(X, y, learning_rate = 0.01, epochs = 1000) {
  n <- nrow(X)
  m <- ncol(X)
  k <- ncol(y)
  
  # Инициализация весов
  W <- matrix(rnorm(m * k), m, k)
  
  # Обучение модели
  for (i in 1:epochs) {
    X <- as.matrix(X)
    y <- as.matrix(y)
    y_pred <- apply(X %*% W, 1, softmax)
    error <- t(y_pred) - y
    
    # Обновление весов
    W <- W - (learning_rate / n) * t(X) %*% error
  }
  return(W)
}

# Разделение на обучающую и тестовую выборки
train_idx <- sample(1:nrow(iris), size = round(0.7 * nrow(iris)), replace = FALSE)
train_data <- iris[train_idx, ]
test_data <- iris[-train_idx, ]

X_train <- train_data[, -5]
y_train <- model.matrix(~ Species - 1, data = train_data)

X_test <- test_data[, -5]
y_test <- model.matrix(~ Species - 1, data = test_data)

# Изменим датафрейм тестовой выборки
col_numbers <- which(y_test == 1, arr.ind = TRUE)
y_test <- col_numbers[rowSums(y_test) == 1, 2]

# Обучение модели с помощью градиентного спуска
W <- gradient_descent(X_train, y_train)

# Оценка точности модели на тестовой выборке
scores <- exp(as.matrix(X_test) %*% W)
probs <- scores / rowSums(scores)
y_pred <- apply(probs, 1, which.max)
accuracy <- sum(y_pred == y_test) / length(y_test)
cat("Accuracy:", accuracy)



