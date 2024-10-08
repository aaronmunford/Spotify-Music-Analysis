# Spotify Music Analytics: Predicting Popularity and Genre Trends

## Project Overview
This project analyzes a dataset of 52,000 Spotify songs to uncover the factors that contribute to song popularity and the unique audio features that define different genres. The analysis includes data preprocessing, regression models, significance tests, and PCA to predict song popularity and classify music by genre.

## Data Description 
The dataset used in this project is a collection of 52,000 Spotify songs, each with features like:

- **Popularity: Integer from 0 to 100, representing how many times the song was played.
- **Duration: Length of the song in milliseconds.
- **Danceability, Energy, Loudness: Quantitative features from Spotify's API that describe the song's characteristics.
- **Genre: Assigned genre of the song (e.g., "hip-hop", "classical").

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

## Results:
#### 1. Song Duration vs. Popularity
![Song Popularity vs. Duration](images/Song_Duration_vs_Popularity.png)

#### 2. Danceability and Energy Analysis
![Danceability and Energy](images/danceability_vs_energy.png)


