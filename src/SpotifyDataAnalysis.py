#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 00:16:05 2024

@author: aaron
"""





#%% 1a) Exploring the Dataset 

# Importing our libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA #This will allow us to do the PCA efficiently
from PIL import Image #These packages will allow us to do some basic
import requests #imaging of matrices and web scraping
from sklearn.metrics import roc_curve, roc_auc_score

import random

randomState = random.seed(15592173)


df = pd.read_csv('spotify52kData.csv', header=0) #load dataset
missingVals = df.isnull().sum() #check for missing values 
description = df.describe() #get an overview of numerical features 

# Feature list for easier iteration
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

#%% Handling missing vals 
pruningMode = 2

# 2) Row wise: (and showing off isnan, as a different way of doing this)
if pruningMode == 2:
  # Create a boolean mask to identify rows with zeros
  zero_mask = df[features].eq(0).any(axis=1)  # Check for any zeros in each row
  df = df[~zero_mask]  # Select rows where there are no zeros (excluding NaNs)
    
    
#Compute and visualize the correlation matrix
corrMatrix = np.corrcoef(df[features],rowvar=False) #Compute the correlation matrix
# Plot the data:
plt.imshow(corrMatrix) 
plt.xlabel('feature')
plt.ylabel('feature')
plt.colorbar()
plt.show()


#%% 1) Exploring the Dataset 

#check for NaNs
missing_values = df.isnull().sum()
print(missing_values)


# Histogram for duration
plt.hist(df['duration'], bins = 1000)
plt.xlabel("Duration (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Song Duration")
plt.xlim(9000, 650000)
plt.show()

# Histogram for popularity
plt.hist(df['popularity'], bins = 30)
plt.xlabel("popularity")
plt.ylabel("Frequency")
plt.title("Distribution of popularity")
plt.show()

# Histogram for danceability
plt.hist(df['danceability'], bins = 500)
plt.xlabel("Danceability")
plt.ylabel("Frequency")
plt.title("Distribution of Danceability")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for energy
plt.hist(df['energy'], bins = 500)
plt.xlabel("energy")
plt.ylabel("Frequency")
plt.title("Distribution of Energy")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for loudness
plt.hist(df['loudness'], bins = 500)
plt.xlabel("dBs")
plt.ylabel("Frequency")
plt.title("Distribution of Loudness")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for speechiness - Poisson distribution?
plt.hist(df['speechiness'], bins = 500)
plt.xlabel("speechiness")
plt.ylabel("Frequency")
plt.title("Distribution of Speechiness")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for acousticness
plt.hist(df['acousticness'], bins = 200)
plt.xlabel("acousticness")
plt.ylabel("Frequency")
plt.title("Distribution of Acousticness")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for instrumentalness
plt.hist(df['instrumentalness'], bins = 500)
plt.xlabel("instrumentalness")
plt.ylabel("Frequency")
plt.title("Distribution of Instrumentalness")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for liveness
plt.hist(df['liveness'], bins = 500)
plt.xlabel("liveness")
plt.ylabel("Frequency")
plt.title("Distribution of Liveness")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for valence - a lot of songs are strangely negative? 
plt.hist(df['valence'], bins = 200)
plt.xlabel("valence")
plt.ylabel("Frequency")
plt.title("Distribution of Valence")
#plt.xlim(9000, 650000)
plt.show()

# Histogram for tempo - somewhat normal?
plt.hist(df['tempo'], bins = 50)
plt.xlabel("BPM")
plt.ylabel("Frequency")
plt.title("Distribution of Tempo")
plt.xlim(20, 225)
plt.show()



# Create a 2x5 figure for all histograms
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # Adjust figsize for better visualization

# Enumerate through features and create histograms
for i, feature in enumerate(features):
    row, col = divmod(i, 5)  # Calculate row and column for subplot
    axes[row, col].hist(df[feature], bins=200)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel("Frequency")
    axes[row, col].set_title(f"{feature} Distribution")

    # Adjust zoom range for specific features (optional)
    if feature == 'duration':
        axes[row, col].set_xlim(000, 650000)
    elif feature == 'tempo':
        axes[row, col].set_xlim(20, 225)

# Tight layout to avoid overlapping labels
plt.tight_layout()
plt.show()

#%% 2) Correlation - is there a relationship between song length and popularity of a song?

# Select relevant columns
data = df[['duration', 'popularity']]

#Calculate Pearson correlation coefficient
corr_matrix = np.corrcoef(data.values.T)  # Select relevant columns and transpose
r = corr_matrix[0, 1]  # Get correlation between duration (index 0) and popularity (index 1)

#Spearman's rank correlation
rho = stats.spearmanr(df['duration'], df['popularity'])  # Get correlation only
rhoValue = rho.correlation

# Visualize (Make scatterplot):
plt.scatter(df['duration'], df['popularity'])
plt.xlabel("Duration (ms)")
plt.ylabel("Popularity")
plt.title("Song Duration vs Popularity")
plt.xlim(0, 1300000)  # Adjust the upper limit if needed

plt.show()

print("r:", r)
print("Rho:", rhoValue)

#%% 3) Are explicitly rated songs more popular than songs that are not explicit?

# Group songs by explicit rating
explicit_ratings = df[df['explicit'] == True]['popularity']
non_explicit_ratings = df[df['explicit'] == False]['popularity']
#Perform Mann-Whitney U test
u, p = stats.mannwhitneyu(explicit_ratings, non_explicit_ratings)
# Calculate medians
explicit_median = np.median(explicit_ratings)
non_explicit_median = np.median(non_explicit_ratings)
# Determine direction of difference|
if explicit_median > non_explicit_median:
  print("Explicit songs have a higher median popularity.")
else:
  print("Non-explicit songs have a higher median popularity.")
#visulization 

#%% 4) Are songs in major key more popular than songs in minor key?

# Group songs by key
major_key_ratings = df[df['mode'] == 1]['popularity']
minor_key_ratings = df[df['key'] == 0]['popularity']

major_median = np.median(major_key_ratings)
minor_median = np.median(minor_key_ratings)

# Perform Mann-Whitney U test
u, p = stats.mannwhitneyu(major_key_ratings, minor_key_ratings)

#%% 5) Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute)
# that this is the case?

#scatterplot:
plt.scatter(df['loudness'], df['energy'])
plt.xlabel("Loudness (dB)")
plt.ylabel("Energy")
plt.title("Relationship Between Loudness and Energy")
plt.show()
#positve correlation 
corr_matrix = np.corrcoef(df[['loudness', 'energy']].values.T)  # Select relevant columns and transpose
r = corr_matrix[0, 1] #Classical Pearson correlation 
#Spearman's rank correlation
rho = stats.spearmanr(df['loudness'], df['energy'])  # Get correlation only
rhoValue = rho.correlation

#%% 6) Which of the 10 individual (single) song features from question 1 predicts popularity best?
# How good is this “best” model?
target = 'popularity'

# Create an empty dictionary to store correlations
correlations = {}

for feature in features:
  # Calculate correlation coefficient using Pearson correlation
  correlation = df[[feature, 'popularity']].corr().iloc[0, 1]
  correlations[feature] = correlation

# Print the correlations
print("Correlations between features and popularity:")
for feature, correlation in correlations.items():
  print(f"{feature}: {correlation:.2f}")


#Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)

#Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Convert back to DataFrame for easier handling
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)
#Fit a linear regression model for each feature
best_feature = None
best_r2 = -float('inf')  # Initialize to a very low value
best_rmse = float('inf')  # Initialize to a very high value
results = []
for feature in features:
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]
    
    model = LinearRegression()
    model.fit(X_train_feature, y_train)
    
    y_pred = model.predict(X_test_feature)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculate RMSE
    
    results.append((feature, r2, rmse))
    
    if r2 > best_r2:
        best_r2 = r2
        best_rmse = rmse
        best_feature = feature

# Step 4: Determine the best feature
print(f"Best Feature: {best_feature}")
print(f"R-squared Value: {best_r2}")
print(f"Root Mean Squared Error: {best_rmse}")

# Display results for all features
for result in results:
    print(f"Feature: {result[0]}, R-squared: {result[1]}, RMSE: {result[2]}")

#%% 6b) Lasso Regression

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Step 4: Fit a LASSO regression model for each feature
alpha = 0.1  # Regularization strength
best_feature = None
best_r2 = -float('inf')  # Initialize to a very low value
best_mse = float('inf')  # Initialize to a very high value
results = []

for feature in features:
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]
    
    model = Lasso(alpha=alpha)
    model.fit(X_train_feature, y_train)
    
    y_pred = model.predict(X_test_feature)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    results.append((feature, r2, mse))
    
    if r2 > best_r2:
        best_r2 = r2
        best_mse = mse
        best_feature = feature

# Step 5: Determine the best feature
print(f"Best Feature: {best_feature}")
print(f"R-squared Value: {best_r2}")
print(f"Mean Squared Error: {best_mse}")

# Display results for all features
for result in results:
    print(f"Feature: {result[0]}, R-squared: {result[1]}, MSE: {result[2]}")


#%% 7) Building a model that uses *all* of the song features from question 1, how well can you
#predict popularity now? How much (if at all) is this model improved compared to the best
#model in question 6). How do you account for this?

# Split data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randomState)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared Value: {r2}")
print(f"Root Mean Squared Error: {rmse}")

# Access coefficients for interpretation (optional)
coefficients = model.coef_
feature_names = X.columns

# Print coefficients for all features
for feature, coef in zip(feature_names, coefficients):
  print(f"{feature}: {coef:.4f}")


#%% 8) When considering the 10 song features above, how many meaningful principal components
# can you extract? What proportion of the variance do these principal components account for?

#PCA
#z-score the data
zscoredData = stats.zscore(df[features])

# 2. Initialize PCA object and fit to our data:
pca = PCA().fit(zscoredData) #Actually create the PCA object

# 3a. Eigenvalues: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# 3b. "Loadings" (eigenvectors): Weights per factor in terms of the original data.
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing

# 3c. Rotated Data: Simply the transformed data
rotatedData = pca.fit_transform(zscoredData)

# 4. For the purposes of this, you can think of eigenvalues in terms of variance explained:
varExplained = eigVals/sum(eigVals)*100

# Now let's display this for each factor:
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))

# There is a single factor that explains 29% of the data 

#%% 1c) Scree Plot
# What a scree plot is: A bar graph of the sorted Eigenvalues
numfeatures = len(features)
x = np.linspace(1,numfeatures,numfeatures)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numfeatures],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

# based off the Kaiser criterion we should select 3 factors (eigenvalue > 1) 
#%% 1d) Interpreting the factors 

whichPrincipalComponent = 2 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Feature')
plt.ylabel('Loading')
plt.show() # Show bar plot
print(features) # Display questions to remind us what they were

#PC 1 - high energy pop 
#PC 2 - Upbeat and Lively 
#PC 3 - electronic music/club music
#%% 1e) Visualize our data in the new coordinate system

plt.plot(rotatedData[:,0]*-1,rotatedData[:,1]*-1,'o',markersize=1) #Again the -1 is for polarity
#good vs. bad, easy vs. hard
plt.xlabel('Popiness of the song')
plt.ylabel('upbeatness & livliness')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Assuming 'rotatedData' contains your data in the new coordinate system

fig = plt.figure(figsize=(10, 6))  # Adjust figure size as needed
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(rotatedData[:, 0]*-1, rotatedData[:, 1]*-1, rotatedData[:, 2]*-1, s=10, c='blue', marker='o')  # Adjust marker size and color

# Customize labels and title
ax.set_xlabel('Popiness')
ax.set_ylabel('Upbeatness & Liveliness')
ax.set_zlabel('Clubiness')  # Replace with the name of your 3rd feature

# Optional: Add a title
ax.set_title('Songs in Principal Components Space')

# Rotate the plot for better viewing (experiment with different angles)
ax.view_init(elev=15, azim=-60)  # Adjust elevation and azimuth angles

plt.show()

#%% 9)  Can you predict whether a song is in major or minor key from valence? If so, how good is this
#prediction? If not, is there a better predictor? [Suggestion: It might be nice to show the logistic
#regression once you are done building the model]

print(len(df[['valence']]))
print(len(df['mode']))

#logistic regresion model 
X_train1, X_test1, y_train1, y_test1 = train_test_split(df[['valence']], df['mode'], test_size=0.3, random_state=randomState)
# Logistic regression model for key prediction
model = LogisticRegression()
model.fit(X_train1, y_train1)
y_pred1 = model.predict(X_test1)
# Evaluate model performance
accuracy = accuracy_score(y_test1, y_pred1)
precision = precision_score(y_test1, y_pred1)
recall = recall_score(y_test1, y_pred1)
f1 = f1_score(y_test1, y_pred1)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Print model coefficients (optional)
print("Coefficients:", model.coef_)
print("Intercepts:", model.intercept_)

model_coefficents = model.coef_

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test1, y_pred1)
auc = roc_auc_score(y_test1, y_pred1)

#plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Song Key Prediction')
plt.legend(loc="lower right")
plt.show()



# Split data into features (X) and target variable (y)
X = df[['valence', 'acousticness', 'danceability']]  # Include other features if desired
y = df['mode']  # Assuming 'key' is a binary variable (0: minor, 1: major)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build logistic regression models

# Model 1: Using valence only
model_valence = LogisticRegression()
model_valence.fit(X_train[['valence']], y_train)

# Model 2: Using acousticness and danceability
model_multi_features = LogisticRegression()
model_multi_features.fit(X_train, y_train)

# Evaluate model performance (accuracy, precision, recall, ROC AUC)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Evaluate model 1 (valence)
y_pred_valence = model_valence.predict(X_test[['valence']])
accuracy_valence = accuracy_score(y_test, y_pred_valence)
precision_valence = precision_score(y_test, y_pred_valence)
recall_valence = recall_score(y_test, y_pred_valence)
roc_auc_valence = roc_auc_score(y_test, y_pred_valence)

# Evaluate model 2 (multi-features)
y_pred_multi_features = model_multi_features.predict(X_test)
accuracy_multi_features = accuracy_score(y_test, y_pred_multi_features)
precision_multi_features = precision_score(y_test, y_pred_multi_features)
recall_multi_features = recall_score(y_test, y_pred_multi_features)
roc_auc_multi_features = roc_auc_score(y_test, y_pred_multi_features)

# Print evaluation metrics for comparison

print("Model 1 (Valence only):")
print(f"Accuracy: {accuracy_valence:.4f}")
print(f"Precision: {precision_valence:.4f}")
print(f"Recall: {recall_valence:.4f}")
print(f"ROC AUC: {roc_auc_valence:.4f}")

print("\nModel 2 (Acousticness & Danceability):")
print(f"Accuracy: {accuracy_multi_features:.4f}")
print(f"Precision: {precision_multi_features:.4f}")
print(f"Recall: {recall_multi_features:.4f}")
print(f"ROC AUC: {roc_auc_multi_features:.4f}")


#%% 10) Which is a better predictor of whether a song is classical music – duration or the principal
#components you extracted in question 8? [Suggestion: You might have to convert the
#qualitative genre label to a binary numerical label (classical or not)]
# Convert the genre label to binary (classical or not)
df['is_classical'] = df['track_genre'].apply(lambda x: 1 if x == 'classical' else 0)

# Define the target variable
y = df['is_classical']

# Split the data into training and testing sets
X_duration = df[['duration']]
X_pca = rotatedData[:, :3]  # Using the first 3 principal components as per the scree plot analysis

X_train_duration, X_test_duration, y_train, y_test = train_test_split(X_duration, y, test_size=0.3, random_state=randomState)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=randomState)


# Standardize the duration feature
scaler = StandardScaler()
X_train_duration = scaler.fit_transform(X_train_duration)
X_test_duration = scaler.transform(X_test_duration)

# Logistic Regression Model using Duration with class weight
model_duration = LogisticRegression(random_state=randomState, class_weight='balanced')
model_duration.fit(X_train_duration, y_train)
y_pred_duration = model_duration.predict(X_test_duration)
y_pred_proba_duration = model_duration.predict_proba(X_test_duration)[:, 1]

# Logistic Regression Model using PCA components with class weight
model_pca = LogisticRegression(random_state=randomState, class_weight='balanced')
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)
y_pred_proba_pca = model_pca.predict_proba(X_test_pca)[:, 1]

# Evaluate Models
accuracy_duration = accuracy_score(y_test, y_pred_duration)
precision_duration = precision_score(y_test, y_pred_duration)
recall_duration = recall_score(y_test, y_pred_duration)
f1_duration = f1_score(y_test, y_pred_duration)
roc_auc_duration = roc_auc_score(y_test, y_pred_proba_duration)

accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)
roc_auc_pca = roc_auc_score(y_test, y_pred_proba_pca)

# Print results
print("Duration Model Performance:")
print(f"Accuracy: {accuracy_duration:.4f}")
print(f"Precision: {precision_duration:.4f}")
print(f"Recall: {recall_duration:.4f}")
print(f"F1 Score: {f1_duration:.4f}")
print(f"ROC AUC Score: {roc_auc_duration:.4f}")

print("\nPCA Model Performance:")
print(f"Accuracy: {accuracy_pca:.4f}")
print(f"Precision: {precision_pca:.4f}")
print(f"Recall: {recall_pca:.4f}")
print(f"F1 Score: {f1_pca:.4f}")
print(f"ROC AUC Score: {roc_auc_pca:.4f}")

# Compare models
if roc_auc_pca > roc_auc_duration:
    print("\nThe principal components are a better predictor of whether a song is classical music.")
else:
    print("\nDuration is a better predictor of whether a song is classical music.")


#%% Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of
#an answer (implied or explicitly) to these enumerated questions [Suggestion: Do something with the
#number of beats per measure, something with the key, or something with the song or album titles]

# Group by track_genre and calculate average tempo
average_tempo_per_genre = df.groupby('track_genre')['tempo'].mean()

# Sort in descending order
average_tempo_per_genre = average_tempo_per_genre.sort_values(ascending=False)

# Print the average tempo for each track_genre sorted in descending order
print("Average Tempo per Track Genre (Descending Order):")
print(average_tempo_per_genre)