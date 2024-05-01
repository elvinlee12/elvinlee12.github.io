## About Me

Hello, I am a Finance Major at Lehigh, graduating May 2025. 
I am from Los Angeles, CA, and I was born in Korea.
My interests include sports, poker, and exercising.


<!-- Upload your own photo and change the path -->

<p style="text-align:center;">
  <img class="img-circle" src="images/headshot.png" width="50%">
</p>

---

## Portfolio

_**[Sentiment Analysis on 10-Ks using NLP to understand stock prices](report/report.md)**_

Used NLP to do a sentiment analysis on certain words by comparing to 2022 stock returns. 
Focused on the sentiments of three topics: Macro-economic, growth, and risk.

Below shows the scatter-plot of negative growth words relating to stock returns three days after the 10k release. 

<img src="report/output_22_22.png?raw=true"/>

---

_**[Prediction Contest](prediction_contest/modelcode.md)**_

The goal of this contest was to generate a model using that predicts the sale price of housing using various variables. The contest was then scored based on the r2 of the predicted sale price of the holdout set to the actual sale price. 

To begin, I started by testing several different ml models to determine which ones would be most optimal in creating a model for sale price. 

The tested modes were: 

Sklearn:
- Linear Regression
- Ridge
- RandomForestRegressor
- GradientBoostingRegressor
  
XGBOOST:
- XGBRegressor

The below scatter plot shows the different types of models and how they performed on the train set. 

<img src="prediction_contest/output_10_0.png?raw=true"/>


From this, XGBoost and the GradientBoostingRegressors showed the best results. XGBoost is a type of gradient boosting, and I decided to use XGBoost due to its advanced nature. 

Using XGBoost as my model, I was able to optomize certain hyperparameters, and achieved an r2 on the holdout set of ~0.88

__In the competetion among my classmates, I placed first!__

---

_**[Swish Analytics](https://brandon4106.github.io/Fin_377_Swish_Insights/)**_

For our Final Project in Fin 377, our team (Brandon Smeltz, Michael Parker, and myself) worked to create a machine learning model to predict the correct points spread on a given NBA Team.

We focused our research on the Boston Celtics during the 2023-2024 NBA season. Our training data was the data during the dates 11/01/23 - 02/29/24, and our testing data was the entire month of march. The test will be betting 100 either on the spread of the Celtics or their opponent for every game they played in March. 

A custom scorer was implemented to maximize profits while looking at the odds given for each line: 

```
def custom_profit_score(y, y_pred, celtics_line, celtics_payout, opp_payout, bet=None):
    if bet is None:
        bet = np.ones(len(y))

    if type(bet) in [int, float]:    
        bet = np.ones(len(y)) * bet
    
    bet_on_celtics = y_pred > (celtics_line * -1)
    celtics_win = y > (celtics_line * -1)
    opponent_win = y < (celtics_line * -1)

    payout = ((bet_on_celtics == celtics_win) * (((100/(celtics_payout*-1))*bet*(bet_on_celtics))+((100/(opp_payout*-1))*bet*(1-bet_on_celtics)))) + ((bet_on_celtics == celtics_win) * bet)

    return(sum(payout) - sum(bet))
```

Our best model was created using the Ridge model, with feature selction (SelectKBest()).

From this model, we optimized certain hyper parameters, with our best pipe being: 
```
best_pipe = Pipeline(
    [
        ('preproc',preproc_pipe),
        ('feature_select',SelectKBest(k=8)),
        ('clf',Ridge(alpha=2.682696))
    ])
```

Next, this pipe was fit, and when tested on our holdout set, we found a profit of $500. 

Below shows the predicted y value in comparison to the actual results:
<img src="images/model_three.png?raw=true"/>

---

_**[Some personal project](/pdf/sample_presentation.pdf)**_

<img src="images/dummy_thumbnail.jpg?raw=true"/>

---

## Career Objectives

I am a driven Finance student with the aspirations of breaking into Investment Banking. 
I have experience creating financial models, and have a comprehensive understanding of DCFs and LBOs. 
My resume is attached below. 
[Resume](images/resume.pdf)

---

## Hobbies

Maybe include a little about these, especially if they are the kinds of things that work well in interviews.

---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
