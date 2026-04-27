<!-- ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)-->
<!-- ![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)-->

# Letterboxd Machine Learning (LML)

## Abstract
Using web-scraping techniques on Letterboxd's 50 most popular users, we build a metadata database of 32,803 movies and extract individual ratings of users. We predict absolute rating, relative ratings and like probability using CatBoost ML and Ridge stacking on user ratings, movie metadata and platform-level signals. Our model achieves a 0.467 R², a 0.561 points mean error and a 30.2% hit rate across 127,535 observations. Accuracy improves to 54.7% and 84.4% within ±0.5 and ±1.0 stars. Results suggest that prediction quality depends less on sample size than on user rating style, with more structured rating distributions and lower excess kurtosis associated with better performance. Measurable variables and social reception of movies can be used to estimate user tastes, but limitations of this model underline that outcome remains affected by unobserved variables such as individual sentiment.

<!-- Your Letterboxd watchlist is too long? No idea of which movie to pick for tonight? Don't worry, use this machine learning algorithm to artificially replicate your cinematographic taste in order to choose the perfect statistical match! No passion, only numbers. -->

## Introduction

[Letterboxd](https://letterboxd.com) is an online social network for movie enthusiasts. Users can keep track of movies they've seen and rate them on a 0.5 to 5 stars scale with the option to "like" a movie: a binary independent variable. <!--When a user adds any movie to its top-four all-time favorites, he becomes a "fan" of said movie. Finally, users can save movies they want to see in their "watchlist". --> All of this data is transparent: rating distribution and user reviews are accessible to anyone online.

<!-- **Disclaimer**: Rated movies have a lot of varied features. On a small sample, this algorithm will struggle to learn anything different than the mean rating. To ensure good performance and algorithmic confidence for high and low ratings, a profile with consistent ratings of 1000+ movies is recommended. In this example, we obtain high performances using [Karsten's Letterboxd profile](https://letterboxd.com/Kurstboy/) (2225 rated movies). -->

## I. Web-scraping

```scraper_iterate.py``` uses Selenium and Undetected ChromeDriver to bypass Cloudflare detection to scrape names, URLs, and respective ratings of all rated movies of a given user. When loading these URLs in an automated Chrome browser, we collect all relevant metadata available on Letterboxd pages:
- **Film attributes**: ```title```, ```year```, ```duration_mins```, ```genre```, ```actors```, ```directors```, ```writer```, ```producers```, ```studios```, ```country```, ```primary_language```
- **Platform metrics**: ```views```, ```likes```, ```fans```, ```avg_rating```, ```total_ratings```, ```rating_count``` per rating level

We also compute personalized metrics, such as: 
- ```rating_std_dev```
- ```rating_ratio``` (ratings to view ratio)
- ```like_view_ratio``` (like-to-view ratio)

We only fetch the first five actors and the main writer: having too many individuals doesn't improve model performance.

With one active browser, metadata scraping is time-consuming (~1.3 sec/movie) since efficiency is limited by Cloudflare. While we could use TMDB's API to instantly fetch movie metadata, the slow Letterboxd scraping is mandatory to obtain ratings and likes given by users of the platform. To avoid scraping the same movie twice, we distinguish individual rating files per user from global movie metadata stored in the ```out/movies.csv``` file. The more users are added to the database, the fewer new movies need to be scraped. 

Our sample focuses on the [50 most popular Letterboxd users](https://letterboxd.com/members/popular/): official or amateur cinema critics, writers, podcasters, or influencers. Popularity is defined as the total amount of likes obtained by a user on its film reviews. Our sample has a combined count of 2,626,328 followers and 128,857 rated movies. Most of these movies overlap: our ```movies.csv``` contains 32,803 individual entries **[Figure 1]**.

**Figure 1: Sample characteristics**
```
N=50 users, 127535 observations

                           Total        Mean      Median         Std         Min         Max
Rated movies             128,857    2577.140    2094.000    1850.418         313      10,168
Observations             127,535    2550.700    2087.500    1820.317         312      10,086
Followers              2,626,328   52526.560   29635.000   54900.525       2,854     245,138
```

## II. Machine Learning Models

In ```ML.py```, we first engineer custom features:
- rating skewness
- user average rating per genre, director, writer and actor
- a binary "niche" factor for movies with less than 500 reviews
- a controversy indicator ($RatingSD \times TotalRatings$).

For user averages, we use a leave-one-out (LOO) mechanism to avoid data leakage. For large scale features (views, likes, fans, ratings), we use log-scaling. For hierarchical textual values, such as `director_1`, `director_2` and `director_3`, we discard ranking by turning them into "bag of words", giving more flexibility to the model. Custom features, such as $Director + Writer$ or $Genre + Country$ combos, are too scarce and not optimal for our sample scale. We discarded low-performing features (themes, technical crew, alternative titles, releases data). We do not use sentiment analysis of top-reviews, neither do we use keyword analysis of movie synopsis.

We use an 80/20 split for training and testing samples stratified on ```user_like```, combined with a 5-fold cross-validation per user. We use We use Ridge stacking to combine the outputs of multiple ML models on different targets: stacking can, and does, outperform the prediction performance of each individual model.
1. **Absolute model** seeks the precise user rating (e.g. 3.562)
2. **Delta model** defines the difference between the user rating and the average rating (e.g. +1.679)
3. **Binary Like model** predicts the probability of a movie being liked by the user (e.g. 0.879)

We use CatBoost instead of XGBoost for its improved performance on small, mixed-type tabular data such as movie metadata. We use the following model parameters: 2000 iterations, 0.005 learning rate, 6 depth, and MAE loss for the delta and absolute models (the heavy penalty of RMSE on outliers pushed towards mean-reverting behavior), with custom weights. We tried adaptive equally weighted and exponentially weighted versions, but a hard-coded strict weighting on extreme values performs better for the delta and absolute models. For the like model, we use adaptive weighting depending on the rated-movies-to-liked-movies ratio. After iteration on all 50 users, the summary per user is exported to ```out/performance.csv``` and each rating forecast is exported to ```out/observations.csv```.

## III. Overall results

To assess the performance of our model, we use user-weighted and observation-weighted averages **[Figure 2]**. Across 127,535 rating forecasts, we obtain an average 0.467 R² (above 0.40 for 66% of the sample and above 0.60 for 18% of the sample). Our hit rate - when the continuous prediction falls within the discrete rating increment - is relatively low at around 30%, and only surpasses 50% when allowing for a ±0.5 rating error. The ```std_diff``` variable measures the difference between the observed user-rating standard deviation and the predicted rating standard deviation. Our model is most often more conservative than observed ratings: a lower ```std_diff``` indicates more confidence in its predictions, while a high value denotes mean-reverting behavior. **Figure 3** precisely underlines this result, showing higher performance associated with a lower standard-deviation difference.

**Figure 2: Textual results**
```
Performance aggregate weighted by user (N=50 users, 127535 observations)

                            Mean      Median         Std         Min         Max
R²                         0.470       0.492       0.161       0.014       0.717
Mean error                 0.556       0.549       0.113       0.222       0.738
Hit rate                   31.1%       30.2%        8.8%       20.6%       64.8%
Hit Rate ±0.5              55.8%       55.1%       11.2%       40.7%       98.4%
Hit Rate ±1.0              84.7%       85.4%        6.9%       72.2%       99.5%
std_diff                   0.316       0.303       0.105       0.176       0.582

Performance aggregate weighted by observations (N=50 users, 127535 observations)

                    Weighted mean      Median         Std         Min         Max
R²                         0.467       0.492       0.161       0.014       0.717
Mean error                 0.561       0.549       0.113       0.222       0.738
Hit rate                   30.2%       30.2%        8.8%       20.6%       64.8%
Hit Rate ±0.5              54.7%       55.1%       11.2%       40.7%       98.4%
Hit Rate ±1.0              84.4%       85.4%        6.9%       72.2%       99.5%
std_diff                   0.323       0.303       0.105       0.176       0.582
```

**Figure 3: Performance by standard deviation gap**
<p align="center">
  <img src=out/figures/performance_std_diff.svg width="600">
</p>

In **Figure 4**, we observe that neither performance nor hit rate increases with sample size, indicating that our model performance does not necessarily improve with scale, but relies on other factors, such as user rating style or consistency.

**Figure 4: R², mean error and hit rate by sample size**
<p align="center">
  <img src=out/figures/performance_sample_size.svg width="350">
  <img src=out/figures/hit_rates_by_sample_size.svg width="350">
</p>

To test this hypothesis, we compare user rating-distribution metrics, such as skewness, kurtosis, extreme frequency $(len(0.5)+len(5.0))/len(ratings)$, and entropy **[Figure 5]**. We notice that negative excess kurtosis and high entropy are correlated with higher R². However, the relative frequency of extreme ratings and the skewness of ratings do not play a significant role in improving the model's performance.

**Figure 5: User rating distribution characteristics and performance**
<p align="center">
  <img src=out/figures/user_distribution_metrics_vs_r2.svg width="600">
</p>

With a violin parity plot **[Figure 6]**, we notice the positive trend in predicted ratings, which consistently overestimates low ratings and underestimates high ratings, reflecting the behavior of a rather mean-reverting model. It is interesting to note the shorter tails for the 3.5 and 4.0 ratings, which are to be expected since these are among the most frequent ratings assigned by users in our sample **[Figure 7]**. It is however surprising to observe such large tails for the 3.0 ratings, which is the second most frequent rating in the dataset, and the rather short tails of the 4.5 rating, which is almost as frequent as the 2.5 and 5.0 ratings. This once again suggests that frequency does not correlate with accuracy, but rather with rating type: "good" (3.5, 4.0) and "really good" (4.5) movies seem easier to predict than "excellent" (5.0) or "terrible" ones (0.5, 1.0, 1.5).

**Figure 6: Violin parity plot**
<p align="center">
  <img src=out/figures/observations_parity_plot.svg width="600">
</p>

**Figure 7: Sample observed rating distribution**
<p align="center">
  <img src=out/figures/observations_observed_predicted_histogram.svg width="600">
</p>

**Figure 8** further emphasizes this mean-reverting behavior of our model with a residual analysis that reflects our previous finding. Overall, the residual distribution is slightly negatively skewed.

**Figure 8: Residuals analysis**
<p align="center">
  <img src=out/figures/observations_residuals_by_observed.svg width="350">
  <img src=out/figures/observations_residual_distribution.svg width="350">
</p>

To assess the accuracy of our like model, which outputs a like likelihood between 0 and 1, we compute the F1 score of the model per user, with like thresholds varying from 0.99 to 0.90 **[Figure 9]**. Overall, our model obtains low F1 scores except for outliers, with noticeable improvement as the threshold is lowered.

**Figure 9: Like model F1 per threshold**
<p align="center">
  <img src=out/figures/distribution_F1.svg width="600">
</p>

## IV. Karsten's Example

With 3,976 hours of watchtime, the number one most popular Letterboxd user is a perfect case study. Using `user_overview.py`, we obtain a brief visual summary of his movie-watching habits **[Figure 10]**. **Figure 11** underlines the correlation between the numeric variables used by the model for this user.

**Figure 10: Karsten's rated movies overview**
<p align="center">
  <img src=out/art/overview_kurstboy.svg width="800">
</p>

**Figure 11: Karsten's numerical features correlation**
<p align="center">
  <img src=out/art/kurstboy_correlation_hm.png width="600">
</p>

**Figure 12** displays missing data in our dataset. We notice that no crucial data, such as `avg_rating`, is missing. In fact, Letterboxd does not compute this metric for low-activity movies: we had to manually calculate the weighted average for these films with fewer ratings to obtain an exploitable dataset. Data cleaning is, as always, a crucial step; this graph allows us to spot missing features that need to be manually obtained.

**Figure 12: Karsten's missing features heatmap**
<p align="center">
  <img src=out/art/kurstboy_missing_features.png width="600">
</p>

After fitting our models, these are the results of the stack **[Figure 13]**: a 0.5911 out-of-sample R-squared with a 0.4851 mean error. Taking into account the granularity of the ratings, 33.48% of 448 out-of-sample ratings fall into the expected range: an acceptable result for a model trained on human sentiment-driven data. The *Relative model* has an average 0.7539 weight in the stack: Karsten's ratings can be defined as some sort of $f(Global Public Ratings)$ function.

**Figure 13: Average out-of-sample performance of Ridge stacking (80/20 data split, 5-fold cross-validation)**
```
         Observed   Estimated
count  448.000000  448.000000          Mean error: 0.4851 points
mean     3.618304    3.659575          R2        : 0.5911
std      0.994088    0.805580  
min      0.500000    0.728443          Model weights
25%      3.000000    3.315550          Absolute: 0.2952 
50%      4.000000    3.906414          Relative: 0.7539
75%      4.500000    4.227933          Like    : 0.2705
max      5.000000    4.856194

           Exact    ±0.5    ±1.0
Hits         150     277     403
Misses       298     171      45
Hit rate  33.48%  61.83%  89.96%

Rating  :  0.5     1.0    1.5    2.0    2.5    3.0    3.5    4.0    4.5   5.0
Hit rate:  25.0%  16.7%  10.0%  25.0%  21.7%  26.9%  26.4%  54.5%  48.7%  0.0%
```

Here are the SHAP values graphs of the *Absolute* and *Delta* models **[Figure 14]**. We observe that public ratings, movie genre, and director dynamics play an important role in defining this user's preferences. Precise weights of the top features can be found in **Figure 15**.

**Figure 14: SHAP values for absolute and delta models on Karsten's rated movies**
<p align="center">
  <img src=out/art/kurstboy_SHAP_absolute.png width="350">
  <img src=out/art/kurstboy_SHAP_delta.png width="350">
</p>


**Figure 15: Ten major features importance for absolute and delta models**
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

For model accuracy, we focus on the minimum and maximum difference between observed and estimated values **[Figure 16]**. On the right are those for which our predictions were accurate; on the left are the movies for which Karsten's ratings are, according to our model, inconsistent with his previous notes for similar films. When plotting the distribution of observed and estimated ratings **[Figure 17] [Figure 18]**, we notice that our model 1. manages to be extremely pessimistic for certain movies, 2. is overly optimistic for movies in the 1 to 3.5 range, and 3. is not optimistic enough for "excellent" movies in the 4.5 to 5 range. This can be explained by the -0.95 skewness of Karsten's ratings: this user has a noticeable bias towards high ratings. The model hence manages to understand why this user sometimes, rarely, really dislikes certain movies, but struggles to identify what distinguishes a 4.5/5 movie from one that is "worth" half a point more.

**Figure 16: Accuracy of 10 best and worst estimations**
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
```

**Figure 17: Estimated vs observed ratings**
<p align="center">
  <img src=out/art/kurstboy_ObservedEstimated.svg width="600">
</p>

**Figure 18: Observed vs estimated ratings**
<p align="center">
  <img src=out/art/kurstboy_ObservedEstimated2.svg width="600">
</p>

Finally, the crucial answer. Out of the 527 movies listed in Karsten's watchlist, he can pick between *No Half Measures* (2013), *Come and See* (1985), *A Brighter Summer Day* (1991), *The Cranes Are Flying* (1957), *Nobody Knows* (2004), *The Ascent* (1977), *Fanny and Alexandre* (1982), *As I Was Moving Ahead* (2000), *Sansho the Bailiff* (1954) or *The Tatami Galaxy* (2010) if he wants to ensure an agreeable evening. On the other hand, he should imperatively remove *Crocodile Dundee in Los Angeles* (2001), *Dolittle* (2020), *The Birth of a Nation* (1915), *Ed* (1996), *Space Chimps* (2008), *Nature in the Wrong* (1933), *Bum Voyage* (1934), *Sealskins* (1932), *Quiver* (2018) and *Monkey in the Middle* (2014) from his watchlist - these were probably misclicks according to what he usually likes to watch **[Figure 19]**. However, he should watch them if his goal is to improve the performance of our algorithm: his ratings on these movies will further help our models to understand what features makes a good(bad) movie according to this user.

*Note: Negative predicted ratings for the "worst" movies are domain violations, a byproduct of the unconstrained Ridge stacking and the important weight of the Delta model; they should be interpreted as "strongest recommendations to avoid" and they concern only 5 out of 527 movies.*

**Figure 19: Watchlist Analysis: Predicted Rating and Like Probability**
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
