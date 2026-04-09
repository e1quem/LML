
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)

# Letterboxd Machine Learning (LML)

Don't know which movie to chose from your Letterboxd watchlist for tonight? Don't worry, use this machine learning repo to lose time artificially replicating your taste to pick the perfect statistical match! No passion, only numbers.

**Disclaimer**: Movies have a lot of varied features. A machine learning model will struggle to learn on a small sample. To ensure good performance, consistent rating of 1000+ movies is recommended. In this example, we obtain high performances using [Karsten's Letterboxd profile](https://letterboxd.com/Kurstboy/) (2225 rated movies).

## I. Web-scraping

We use *Selenium* and *Undedected ChromeDriver* to bypass Cloudfare detection in order to obtain a user's rated movies as well as the ratings and likes he assigned to them ('user_scraper.py').  We use the same method to obtain the titles and URLs of all movies contained in a user's watchlist ('watchlist_scraper.py').

'movie_scraper.py' uses each of these CSV files to load individual Letterboxd pages of these movies, collecting rating distribution, crew and other available metadata. For actors, we only take into account the five first ones: having too many actors doesn't improve performance. We also compute personalized metrics: rating standard deviation, ratios, etc. With only one browser, this is a time consuming process (~20min/1000 movies) that could be optimized using multiple workers, at the risk of triggering Cloudfare's firewall.

## II. Machine Learning Models

We first compute custom features: global rating skewness of a movie, user average per genre, director, writer, actor, etc. For these user averages, we use a leave-one-out mechanism in order to avoid data-leakage and overfitting. We turn hierarchical textual values, such as director_1, _2 and _3, into "bag of words", in order to give more flexibility for the model to converge.

We use a 80/20 data split for training and testing. After trial and error, we use multiple models at once, on different targets:
1. The Asbolute model tries to find the precise user rating.
2. The Delta model tries to define the delta between the average rating and the user rating.
3. The Binary like model tries to define the odds of a movie being liked by the user.

We use CatBoost instead of XGBoost because it performs better on small datasets such as rated movies lists. CatBoost model parameters, such as weight on extreme ratings (we tried an equally weighted version, an exponentially weighted version, but a hard coded strict weighting on extreme values performs better), learning rates, depth, loss function, iterations, are those that we found to perform the best on a general basis. Users are free to figure out what works best for their dataset. Notably, the Like weight needs to be adjusted according to the liked movies/total rated movies ratio of a dataset.

We use ridge stacking to combine the three models.

## III. Karsten's Example

With 3955.2 hours of watchtime, his famous profile is a prime case study. Using `overview.py`, we obtain a brief visual analysis of his movie-watching habits [Figure 1].

**Figure 1: Karsten's Rated Movies Overview**
<p align="center">
  <img src=out/art/overview.svg width="600">
</p>

This heatmap [Figure 2] displays missing data in our dataset. We notice that no crucial data, such as avg_rating, is missing. In fact, Letterboxd doesn't compute this metric for low-activity movies: we had to manually calculate the weighted average for these few films to obtain an exploitable dataset. Data cleaning is as always a crucial step, and this graph allows you to spot missing features to need to be manually obtained.

**Figure 2: Karsten's Missing Features Heatmap**
<p align="center">
  <img src=out/art/missing_features.png width="500">
</p>

After fitting the models, these are the results: a surprisingly high 0.61 R-squared on the out-of-sample performance. The Relative model plays a major role in this result: Karsten's ratings can be defined as a function of the global public rating. 

```
         Observed   Estimated
count  445.000000  445.000000
mean     3.561798    3.561798
std      1.066537    0.831410
min      0.500000    0.445751
25%      3.000000    3.116830
50%      4.000000    3.797880
75%      4.500000    4.171409
max      5.000000    5.007117

Mean error: 0.5115 points.
R2 : 0.6116

Model weights
Absolute: 0.1115
Relative: 0.9514
Like    : -0.2481
```

Here are the SHAP values graph of the Absolute and Delta models [Figure 3]. We observe that rating, genre and director dynamics play an important role in defining his preferences. Precise weights can be found in Figure 4, and SHAP dependence plots of top features can be found in Figure 5 and Figure 6.

**Figure 3: SHAP values for Absolute and Delta models on Karsten's Rated Movies**
<p align="center">
  <img src=out/art/SHAPabsolute.png width="300">
  <img src=out/art/SHAPdelta.png width="300">
</p>


**Figure 4: Ten Major Feature Importance for Absolute and Delta Models**
```
Feature importance for Absolute model (top 10)         Feature importance for Delta model (top 10):
rating_skew    : 27.61                                 actors: 16.40
like_view_ratio: 22.43                                 rating_skew: 12.36
avg_rating     : 19.99                                 like_view_ratio: 10.65
user_genre_avg : 7.17                                  user_genre_avg: 7.53
genre_1        : 4.56                                  rating_ratio: 6.39
actors         : 4.38                                  studios: 5.94
rating_std_dev : 3.89                                  rating_std_dev: 5.71
studios        : 1.54                                  genre_1: 5.01
producers      : 1.45                                  user_dir_avg: 4.65
user_dir_avg   : 1.20                                  avg_rating: 3.64
```

**Figure 5: SHAP Dependence for Absolute Model Major Feature**
<p align="center">
  <img src=out/art/Absolute_avg_rating_SHAP_dependence.svg width="225">
  <img src=out/art/Absolute_like_view_ratio_SHAP_dependence.svg width="225">
  <img src=out/art/Absolute_rating_skew_SHAP_dependence.svg width="225">
</p>

**Figure 6: SHAP Dependence for Delta Model Major Feature**
<p align="center">
  <img src=out/art/Delta_like_view_ratio_SHAP_dependence.svg width="250">
  <img src=out/art/Delta_rating_skew_SHAP_dependence.svg width="250">
</p>

For model accuracy, we can look at the minimum and maximum difference between obsered and estimated values in Figure 7. On the right are those for which our forecast was accurate; on the left are the movies for which Karsten's rating is inconsistent with his previous ratings for similar films according to our model. Repartition of observed and estimated ratings can be found in Figure 8 and 9. We notice that our model manages to be extremely pessimistic for certain movies (bad notes are rare compared to high, the dataset ratings have a -0.95 skewness) but isn't optimistic enough for "good" movies.

**Figure 7: Accuracy of 20 Best and Worse Estimations**
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
  <img src=out/art/ObservedEstimated.svg width="600">
</p>

**Figure 9: Observed vs Estimated Ratings**
<p align="center">
  <img src=out/art/ObservedEstimated2.svg width="600">
</p>

Finally, Figure 10 underlines that Karsten out of the 527 movies in his watchlist, can pick between No Half Measures (2013), Come and See (1985), A Britghter Summer Day (1991), The Cranes Are Flying (1957), Nobody Knows (2004), The Ascent (1977), Fanny and Alexandre (1982), As I Was Moving Ahead (2000), Sansho the Bailiff (1954) or The Tatami Galaxy (2010) if he wants to ensure an agreable evening. On the other hand, he should imperatively remove Crocodile Dundee in Los Angeles (2001) (how did this one ended up here?), Dolittle (2020), The Birth of a Nation (1915), Ed (1996), Space Chimps (2008), Nature in the Wrong (1933), Bum Voyage (1934), Sealskins (1932), Quiver (2018) and Monkey in the Middle (2014) from his watchlist - there were probably missclicks according to what he usually likes to watch. In reality, he should watch them anyways: bad notes improve the model understanding of what makes a good(bad) movie for the user.

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