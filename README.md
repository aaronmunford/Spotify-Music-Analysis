# Spotify Music Analytics: Predicting Popularity and Genre Trends

## Project Overview
This project analyzes a dataset of 52,000 Spotify songs to uncover the factors that contribute to song popularity and the unique audio features that define different genres. The analysis includes data preprocessing, regression models, significance tests, and PCA to predict song popularity and classify music by genre.

## Data Description 
The dataset used in this project is a collection of 52,000 Spotify songs, each with features like:

- **Popularity**: Integer from 0 to 100, representing how many times the song was played.
- **Duration**: Length of the song in milliseconds.
- **Danceability, Energy, Loudness**: Quantitative features from Spotify's API that describe the song's characteristics.
- **Genre**: Assigned genre of the song (e.g., "hip-hop", "classical").

The dataset is available in the data/ folder as spotify52kData.csv.
## Key Findings:
- **Song Popularity vs. Duration**: Analyzed the correlation between song duration and popularity. The results show a weak negative correlation, with longer songs tending to be slightly less popular.

- **Explicit Songs**: Compared the popularity of explicit songs vs. non-explicit songs using a Mann-Whitney U test. Explicit songs were found to be significantly more popular.

- **Major vs. Minor Key Popularity**: Compared the popularity of songs in major and minor keys. The analysis found no statistically significant difference in popularity.

- **Energy vs. Loudness**: Strong positive correlation between energy and loudness, confirming that louder songs tend to have higher energy.

- **Principal Component Analysis (PCA)**: Reduced the dimensionality of the dataset to three principal components, explaining 56% of the variance in the data.


## Technologies:
* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

## Results and Key Findings

### Data Preparation
Data was cleaned by removing rows with missing values, and numerical features were standardized. PCA was applied to reduce the dimensionality of the data.

### 1. Distribution of Song Features
Most features showed non-normal distributions, with **danceability** being the closest to normal.
<img src="images/Song%20Feature%20Distributions.png" alt="Distribution of Song Features" width="500"/>

### 2. Song Popularity vs. Duration
There is a weak negative correlation between song duration and popularity, indicating that longer songs are slightly less popular.

<img src="images/Song%20Duration%20vs%20Popularity.png" alt="Song Popularity vs. Duration" width="500"/>

### 3. Explicit Songs Popularity
Explicit songs were found to be significantly more popular than non-explicit songs (p-value < 0.05).

### 4. Major vs. Minor Key Popularity
There was no statistically significant difference in popularity between songs in major and minor keys.

### 5. Energy vs. Loudness
A strong positive correlation was found between energy and loudness, confirming that louder songs tend to have higher energy.
<img src="images/Loudness%20and%20Energy.png" alt="Energy vs. Loudness" width="500"/>

### 6. Best Single Feature for Predicting Popularity
**Instrumentalness** had the highest correlation with popularity (-0.14), but it is a weak predictor overall.

### 7. Multifeature Model for Popularity
Using all 10 features improved the model slightly, with an R-squared of 0.062 and lower prediction error.

### 8. Principal Component Analysis (PCA)
PCA reduced the dataset to 3 components, explaining 56.51% of the variance.
<img src="images/Eigenvalue%20PCA.png" alt="PCA Components" width="500"/>
<img src="images/Component%20Breakdown.png" alt="Component Breakdown" width="500"/>

### 9. Predicting Key from Valence
Valence could moderately predict whether a song is in a major or minor key, with an accuracy of 62.4%.

### 10. Genre Tempo Analysis
Hardstyle and dubstep had much higher tempos compared to genres like chill or ambient music.
<img src="images/BPM%20descending.png" alt="BPM descending" width="500"/>
