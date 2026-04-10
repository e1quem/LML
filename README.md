![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)

# Letterboxd Machine Learning (LML)

Your Letterboxd watchlist is too long? No idea of which movie to pick for tonight? Don't worry, use this machine learning algorithm to artificially replicate your cinematographic taste in order to choose the perfect statistical match! No passion, only numbers.

## Introduction

[Letterboxd](https://letterboxd.com) is an online social network for movie enthusiasts. Users can keep track of movies they've seen and rate them on a 0.5 to 5 stars scale with the option to "like" a movie: a binary independent variable. When a user adds any movie to its top-four all-time favorites, he becomes a "fan" of said movie. Finally, users can save movies they want to see in their "watchlist". All of this data is transparent: rating distribution and user reviews are accessible to anyone.

**Disclaimer**: Rated movies have a lot of varied features. On a small sample, this algorithm will struggle to learn anything different than the mean rating. To ensure good performance and algorithmic confidence for high and low ratings, a profile with consistent ratings of 1000+ movies is recommended. In this example, we obtain high performances using [Karsten's Letterboxd profile](https://letterboxd.com/Kurstboy/) (2225 rated movies).

## I. Web-scraping

We use *Selenium* and *Undetected ChromeDriver* to bypass Cloudflare detection in order to scrape all movies, ratings, likes and watchlist of a given profile (`user_scraper.py`).

`movie_scraper.py` uses these CSV files to load individual Letterboxd movie pages in a Chrome browser, collecting rating distribution, crew and all available metadata. For actors, we only fetch the first five: having too many does not improve model performance. We also compute personalized metrics, such as rating standard deviation, like-to-view ratio and other ratios. With only one active browser, this is the most time-consuming step (~20min/1000 movies). This could be optimized using multiple workers - at the risk of triggering Cloudflare's firewall. For testing purposes, you can find Kursten's preloaded data in the `/out` folder.

*Note: While we could use TMDB API to quickly fetch movies metadata, we would still need to scrape Letterboxd to obtain the ratings and likes given by users of the platform. This is why we obtain all metadata from Letterboxd: it's not the most efficient way, but it's mandatory to go through it.* 

## II. Machine Learning Models

First, we engineer specific features: rating skewness, user average rating per genre/director/writer/actor, niche factor for movies with less than 500 reviews, and a controversy indicator ($RatingSD \times TotalRatings$). For large numbers (views, likes, fans, ratings), we use a log-scale to improve model's performance. For user averages, we use a leave-one-out (LOO) mechanism in order to avoid data-leakage and overfitting. For hierarchical textual values, such as `director_1`, `director_2` and `director_3`, we discard the ranking by turning them into "bag of words" to give more flexibility to the model.

We tried multiple custom features, such as $Director + Writer$ or $Genre + Country$ combos, but such precise variables aren't optimal for small sample sizes. We discarded low-performing features, such as themes, technical crew, alternative titles and releases data. We do not use sentiment analysis of top-reviews, neither do we use keyword analysis of movie synopsis.

We use a 80/20 data split for training and testing samples. After trial and error, we decided to use ridge stacking to combine the outputs of multiple machine learning models on different targets. This way, we do not have to select a single model, while the performance of stacking can - and does - outperform the prediction performance of each individual model.
1. *Absolute model* finds the precise user rating (e.g. 3.562)
2. *Delta model* defines the difference between the user rating and the average rating (e.g. +1.679)
3. *Binary Like model* predicts the probability of a movie being liked by the user (e.g. 0.879)

We use CatBoost instead of XGBoost for its improved performances on small, mixed-type tabular data such as movies metadata. CatBoost model parameters, such as weight on extreme ratings (we tried adaptative equally-weighted and exponentially-weighted versions, but a hard coded strict weighting on extreme values performs better), learning rates, depth, loss function and iterations are those that we found to perform the best on a general basis. Users are free to figure out what works best for their datasets. Notably, the *Binary Like model* weight needs to be adjusted according to the liked-movies/total-rated-movies ratio of a dataset (e.g. if a user has rated 1000 movies and liked 200 of them, the weight should be set to [1,5]).

## III. Karsten's Example

With 3955.2 hours of watchtime, this famous profile is a perfect case study. Using `overview.py`, we obtain a brief visual analysis of his movie-watching habits **[Figure 1]**.

**Figure 1: Karsten's Rated Movies Overview**
<p align="center">
  <img src=out/art/overview.svg width="1000">
</p>

This heatmap **[Figure 2]** displays missing data in our dataset. We notice that no crucial data, such as `avg_rating`, is missing. In fact, Letterboxd doesn't compute this metric for low-activity movies: we had to manually calculate the weighted average for these films with fewer ratings to obtain an exploitable dataset. Data cleaning is as always a crucial step; this graph allows us to spot missing features that need to be manually obtained.

**Figure 2: Karsten's Missing Features Heatmap**
<p align="center">
  <img src=out/art/missing_features.png width="600">
</p>

After fitting our models, these are the results of the stack: a 0.6014 out-of-sample R-squared with a 0.5218 mean error. Taking into account the granularity of the ratings, 31.24% of 445 out-of-sample ratings fall into the expected range: an acceptable result for a model trained on human sentiment-driven data. The *Relative model* has a 0.8582 weight in the stack: Karsten's ratings can be defined as some sort of $f(Global Public Ratings)$ function. 

```
         Observed   Estimated
count  445.000000  445.000000          Mean error: 0.5218 points.
mean     3.561798    3.559830          R2 : 0.6014
std      1.066537    0.834108 
min      0.500000    0.529930          Model weights
25%      3.000000    3.129673          Absolute: -0.0545
50%      4.000000    3.837728          Relative: 0.8582
75%      4.500000    4.179794          Like    : 0.7527
max      5.000000    4.851018 
```

Here are the SHAP values graph of the *Absolute* and *Delta* models **[Figure 3]**. We observe that public ratings, movie genre and director dynamics play an important role in defining this user's preferences. Precise weights can be found in **Figure 4**, and SHAP dependence plots of top features are plotted in **Figure 5** and **Figure 6**.

**Figure 3: SHAP values for Absolute and Delta models on Karsten's Rated Movies**
<p align="center">
  <img src=out/art/SHAPabsolute.png width="400">
  <img src=out/art/SHAPdelta.png width="400">
</p>


**Figure 4: Ten Major Feature Importance for Absolute and Delta Models**
```
Feature importance for Absolute model           Feature importance for Delta model
rating_skew    : 27.61                          actors: 16.40
like_view_ratio: 22.43                          rating_skew: 12.36
avg_rating     : 19.99                          like_view_ratio: 10.65
user_genre_avg : 7.17                           user_genre_avg: 7.53
genre_1        : 4.56                           rating_ratio: 6.39
actors         : 4.38                           studios: 5.94
rating_std_dev : 3.89                           rating_std_dev: 5.71
studios        : 1.54                           genre_1: 5.01
producers      : 1.45                           user_dir_avg: 4.65
user_dir_avg   : 1.20                           avg_rating: 3.64
```

**Figure 5: SHAP Dependence for Absolute Model Major Feature**
<p align="center">
  <img src=out/art/Absolute_avg_rating_SHAP_dependence.svg width="400">
  <img src=out/art/Absolute_rating_skew_SHAP_dependence.svg width="400">
</p>

**Figure 6: SHAP Dependence for Delta Model Major Feature**
<p align="center">
  <img src=out/art/Delta_like_view_ratio_SHAP_dependence.svg width="400">
  <img src=out/art/Delta_rating_skew_SHAP_dependence.svg width="400">
</p>

For model accuracy, we focus on the minimum and maximum difference between observed and estimated values **[Figure 7]**. On the right are those for which our prediction was accurate; on the left are the movies for which Karsten's rating is inconsistent with his previous notes for similar films according to our model. When plotting the distribution of observed and estimated ratings **[Figure 8] [Figure 9]**, we notice that our model *1.* manages to be extremely pessimistic for certain movies, *2.* is overly optimistic for movies in the 1 to 3.5 range and *3.* isn't optimistic enough for "excellent" movies in the 4.5 to 5 range. This can be explained by the -0.95 skewness of Karsten's ratings: this user has a noticeable skew towards high ratings. The model hence manages to understand why this user really dislikes certains movies, but struggles to identify what distinguishes a 4.5/5 from a 5/5 movie.

**Figure 7: Accuracy of 20 Best and WorsT Estimations**
```
min(difference)                               max(difference)
------------------------------------------------------------------------------------------
Chinatown                      -0.007         21                             +2.352
Stop Making Sense              +0.007         The Great Gatsby               +2.269
Three Colours: Red             +0.010         Star Wars: The Last Jedi       -2.218
Inside Out                     -0.011         Superman                       +1.930
Who's That Knocking at My Door +0.012         The Unbearable Weight of Massi +1.867
Icarus                         -0.013         Magic Mike XXL                 -1.751
All About My Mother            +0.013         The Dead Don't Die             -1.730
Glengarry Glen Ross            +0.015         Unfriended                     -1.708
Chip 'n Dale: Rescue Rangers   +0.017         The Lego Movie 2: The Second P -1.694
The Father                     -0.018         Set It Up                      +1.656
Bonnie and Clyde               -0.019         The Chronicles of Narnia: Prin +1.645
The Bride!                     -0.020         Barbie in A Mermaid Tale       +1.590
Shrek the Third                +0.029         The Avengers                   +1.570
Hostel: Part II                +0.032         Last Night in Soho             +1.555
Double Indemnity               +0.032         Chicken Run                    +1.538
FRED: The Movie                +0.033         Star Wars: The Rise of Skywalk -1.535
Escape from the Planet of the  -0.036         King Kong                      -1.485
Rosetta                        +0.037         Tag                            +1.484
A Different Man                +0.040         Jesus 2                        +1.451
Run Lola Run                   -0.040         Green Book                     +1.434
```

**Figure 8: Estimated vs Observed Ratings**
<p align="center">
  <img src=out/art/ObservedEstimated.svg width="700">
</p>

**Figure 9: Observed vs Estimated Ratings**
<p align="center">
  <img src=out/art/ObservedEstimated2.svg width="700">
</p>

Finally, the crucial answer. Out of the 527 movies listed in Karsten's watchlist, he can pick between *No Half Measures* (2013), *Come and See* (1985), *A Brighter Summer Day* (1991), *The Cranes Are Flying* (1957), *Nobody Knows* (2004), *The Ascent* (1977), *Fanny and Alexandre* (1982), *As I Was Moving Ahead* (2000), *Sansho the Bailiff* (1954) or *The Tatami Galaxy* (2010) if he wants to ensure an agreeable evening. On the other hand, he should imperatively remove *Crocodile Dundee in Los Angeles* (2001), *Dolittle* (2020), *The Birth of a Nation* (1915), *Ed* (1996), *Space Chimps* (2008), *Nature in the Wrong* (1933), *Bum Voyage* (1934), *Sealskins* (1932), *Quiver* (2018) and *Monkey in the Middle* (2014) from his watchlist - these were probably misclicks according to what he usually likes to watch **[Figure 10]**. However, he should watch them if his goal is to improve the performances of our algorithm: his ratings on these movies will further help our models to understand what features makes a good(bad) movie according to this user.

*Note: Negative predicted ratings for the "worst" movies are domain violations, a byproduct of the unconstrained Ridge stacking and the important weight of the Delta model; they should be interpreted as "strongest recommendations to avoid" and they concern only 5 out of 527 movies.*

**Figure 10: Watchlist Analysis: Predicted Rating and Like Probability**
```
                               title  Predicted_Rating  Like_Probability
294  No Half Measures: Creating t...          4.866810          0.976227
87                      Come and See          4.701222          0.978141
7              A Brighter Summer Day          4.698863          0.984462
401            The Cranes Are Flying          4.687806          0.984674
296                     Nobody Knows          4.667878          0.990452
390                       The Ascent          4.667034          0.991463
138              Fanny and Alexander          4.660918          0.980444
32   As I Was Moving Ahead, Occas...          4.658658          0.973705
348               Sansho the Bailiff          4.642318          0.991500
466                The Tatami Galaxy          4.635975          0.988743

                               title  Predicted_Rating  Like_Probability
95   Crocodile Dundee in Los Angeles          1.445554          0.107931
115                         Dolittle          1.382716          0.138447
392            The Birth of a Nation          1.369546          0.206615
124                               Ed          1.118659          0.102769
370                     Space Chimps          1.090422          0.060858
289              Nature in the Wrong         -0.821767          0.922104
69                        Bum Voyage         -1.309767          0.635985
350                        Sealskins         -1.519846          0.565092
335                           Quiver         -1.575070          0.326040
271             Monkey in the Middle         -1.611937          0.189666
```