if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(recosystem)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create train and test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Matching userId and movieId in both train and test sets
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)




str(edx)
edx %>% select(-genres) %>% summary()
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(20, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "green", stat = "identity") +
  xlab("Count") +
  ylab(NULL) +
  theme_bw()


edx %>%
  ggplot(aes(rating, y = ..prop..)) +
  geom_bar(color = "black", fill = "green") +
  labs(x = "Ratings", y = "Relative Frequency") +
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) +
  theme_bw()


edx %>% group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", fill = "green", bins = 40) +
  xlab("Ratings") +
  ylab("Users") +
  scale_x_log10() +
  theme_bw()

# Initialise an empty table for storing RMSE results
rmse_table <- data.frame(Method = character(), Model_Name = character())
                         
###Model 1 - Baysesian

# Define Bayesian Average function (same as before)
bayesian_average <- function(x, mu, lambda) {
  prior_mean <- mean(x)
  prior_variance <- var(x)
  posterior_mean <- (lambda * prior_mean + mu * sum(x)) / (lambda + length(x))
  posterior_variance <- (prior_variance / lambda + sum((x - mu)^2)) / (lambda + length(x) - 1)
  return(list(mean = posterior_mean, variance = posterior_variance))
}

# Calculate global average rating (can be adjusted if needed)
global_avg_rating <- mean(edx$rating)

# Tuning parameter for Bayesian Average, adjust as needed
lambda = 10
#results_reco <- 0 had to add this for the knitting to work
# Apply Bayesian Average to predictions
bayesian_predictions <- mapply(bayesian_average, results_reco, MoreArgs = list(mu = global_avg_rating, lambda = lambda), SIMPLIFY = FALSE)
bayesian_predictions <- unlist(lapply(bayesian_predictions, function(x) x$mean))

# Calculate RMSE using Bayesian Average predictions
rmse_bayesian <- sqrt(mean((bayesian_predictions - test_set$rating)^2))
cat("RMSE using Bayesian Average:", rmse_bayesian, "\n")
rmse_table <- rbind(rmse_table, data.frame(Method = "Bayesian Average", Model_Name = "Model 1", RMSE = rmse_bayesian))
cat("Method    Model Name      RMSE\n")
print(rmse_table)


### Model 2: Matrix Factorization with Recosystem
set.seed(1, sample.kind = "Rounding")

# Create Recosystem data objects with descriptive names
train_data_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_data_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))

# Descriptive model name
recommender_model <- Reco()

# Tuning parameters
tuning_results <- recommender_model$tune(train_data_reco, opts = list(dim = c(20, 30),
                                                                      costp_l2 = c(0.01, 0.1),
                                                                      costq_l2 = c(0.01, 0.1),
                                                                      lrate = c(0.01, 0.1),
                                                                      nthread = 4,
                                                                      niter = 10))

# Train the model with descriptive object names
recommender_model$train(train_data_reco, opts = c(tuning_results$min, nthread = 4, niter = 30))

# Generate predictions with descriptive names
matrix_factorization_predictions <- recommender_model$predict(test_data_reco, out_memory())

# Calculate and print RMSE 
matrix_factorization_rmse <- RMSE(matrix_factorization_predictions, test_set$rating)
cat("RMSE using Matrix Factorization with Recosystem:", matrix_factorization_rmse, "\n")
rmse_table <- rbind(rmse_table, data.frame(Method = "Matrix Factorization", Model_Name = "Model 2", RMSE = matrix_factorization_rmse))
cat("\n")  # Add an empty line for visual separation
print(rmse_table)

#  This one has been commented out as am having too many issues to troubleshoot
# ###Model 3 - Collaborative Filtering techniques
# 
# # Function to predict rating for a user-item pair using dplyr
# predict_rating_userbased <- function(user_id, movie_id, edx_data = edx) {
#   # K most similar users (adjust K as needed)
#   k <- 10
#   
#   # Filter ratings excluding the target user
#   user_ratings <- edx_data %>%
#     filter(userId != user_id) %>%
#     select(userId, rating)
#   
#   # Calculate item similarities using dplyr and rowMeans
#   item_similarities <- user_ratings %>%
#     group_by(userId) %>%
#     summarise(similarity = {
#       # Check if ratings exist for the current user
#       if (n() > 1) {
#         rowMeans((rating - mean(rating)) * (edx_data[edx_data$userId == user_id,]$rating - mean(edx_data$rating)))
#       } else {
#         # Handle case with no ratings for the current user (e.g., return 0)
#         0
#       }
#     })
#   # Select top K most similar users
#   neighbors <- item_ratings %>%
#     arrange(desc(similarity)) %>%
#     head(k)
#   
#   # Weighted average rating from similar users for the movie
#   if (nrow(neighbors) > 0) {  # Check if neighbors exist
#     predicted_rating <- sum(neighbors$similarity * (user_ratings[user_ratings$userId %in% neighbors$userId,]$rating)) / 
#       sum(abs(neighbors$similarity))
#   } else {
#     predicted_rating <- mean(edx_data$rating)  # Use mean rating if no neighbors found
#   }
#   return(predicted_rating)
# }
# # Example: Predict rating for user ID 1 and movie ID 10
# user_id <- 1
# movie_id <- 10
# predicted_rating <- predict_rating_userbased(user_id, movie_id)
# cat("Predicted Rating (User-based) for User", user_id, "and Movie", movie_id, ":", predicted_rating, "\n")
# 
# # Calculate RMSE using User-based predictions
# userbased_predictions <- sapply(test_set[, c("userId", "movieId")], function(x) predict_rating_userbased(x[1], x[2], edx = edx))
# userbased_predictions <- unlist(userbased_predictions)
# rmse_userbased <- sqrt(mean((userbased_predictions - test_set$rating)^2))
# 
# # Print RMSE
# cat("RMSE using User-based Collaborative Filtering:", rmse_userbased, "\n")
rmse_table <- rbind(rmse_table, data.frame(Method = "Collaborative Filtering techniques", Model_Name = "Model 3", RMSE = NA))

# Print the table after Model 3
cat("\n")  # Add another empty line
print(rmse_table)


###Model 4 -  Means + Bias
mean_training <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mean_training)
cat("RMSE using Means:", naive_rmse, "\n")

bias <- train_set %>%
        group_by(movieId) %>%
        summarize(bias_i = mean(rating - mean_training))

bias %>% ggplot(aes(bias_i)) +
  geom_histogram(color = "black", fill = "green", bins = 10) +
  xlab("Movie Bias") +
  ylab("Count") +
  theme_bw()


predicted_ratings <- mean_training + test_set %>%
  left_join(bias, by = "movieId") %>%
  pull(bias_i)
mean_bias_rmse <- RMSE(predicted_ratings, test_set$rating)

cat("RMSE using Mean + Bias", mean_bias_rmse, "\n")
rmse_table <- rbind(rmse_table, data.frame(Method = "RMSE using Mean + Bias", Model_Name = "Model 4", RMSE = mean_bias_rmse))
cat("\n")  # Add an empty line for visual separation
print(rmse_table)


###Final Model with testing against test data set
set.seed(1, sample.kind = "Rounding")

# Create Recosystem data objects with descriptive names for edx and validation sets
edx_data_reco <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
validation_data_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))

# Descriptive model name
final_recommender_model <- Reco()

# Tuning parameters
final_tuning_results <- final_recommender_model$tune(edx_data_reco, opts = list(dim = c(20, 30),
                                                                                costp_l2 = c(0.01, 0.1),
                                                                                costq_l2 = c(0.01, 0.1),
                                                                                lrate = c(0.01, 0.1),
                                                                                nthread = 4,
                                                                                niter = 10))

# Train the final model with descriptive object names
final_recommender_model$train(edx_data_reco, opts = c(final_tuning_results$min, nthread = 4, niter = 30))

# Generate final predictions with descriptive names
final_validation_predictions <- final_recommender_model$predict(validation_data_reco, out_memory())

# Calculate and print final RMSE 
final_validation_rmse <- RMSE(final_validation_predictions, validation$rating)
cat("RMSE using Matrix Factorization with Recosystem on Validation Set:", final_validation_rmse, "\n")

rmse_table <- rbind(rmse_table, data.frame(Method = "Matrix Factorization (Validation)", Model_Name = "Final Validation", RMSE = final_validation_rmse))

# Print the final table with all results
cat("\n")  # Add another empty line
print(rmse_table)


####### PRedicting with final_holdout_test
final_holdout_test_reco <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))

final_holdout_predictions <- final_recommender_model$predict(final_holdout_test_reco, out_memory())
final_holdout_rmse <- RMSE(final_holdout_predictions, final_holdout_test$rating)
cat("RMSE on Final Holdout Set:", final_holdout_rmse, "\n")

rmse_table <- rbind(rmse_table, data.frame(Method = "Matrix Factorization (Holdout Set)", Model_Name = "Final Validation", RMSE = final_holdout_rmse))
print(rmse_table)
rmse_table <- rbind(rmse_table, data.frame(Method = "Matrix Factorization (Holdout Set)", Model_Name = "Final Validation", RMSE = final_holdout_rmse))
