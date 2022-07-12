# Sports_Betting_Prediction

The objective of this project is to predict the outcome of all men's professional tennis (ATP) matches that have taken place over the past two decades, and to compare our results. 

The dataset used was empty of interesting and exploitable variables so a lot of the work was dedicated to cleaning and implementing new variables such as ATP ranking, number of wins on different surfaces, head-to-head matches between 2 players and many others. 

Afterwards, several Machine Learning models were tested and this work led to an accuracy of about 65% against 67% for the best bookmaker. 
An additional study was also carried out on the accuracy obtained according to different parameters: the tournaments, the type of tournaments, the surface in order to find the parameters that allow us to maximize our chances of winning our bet. See the report_project.pdf in the github for more details.

This work is still in progress and several tracks are still to explore: 

* Neural networks can still be implemented to maximize accuracy.
* Relevant variables can still be added to the model.
* A deeper study can be done on the probabilities that players have to win a match and compare them to the odds given by the bookmakers.
* A study can be made on the type of matches (depending on the parameters) that can allow us to obtain a higher accuracy than the bookmakers.
