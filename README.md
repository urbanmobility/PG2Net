# PG2Net
PG2Net:Personalized and Group Preference Guided Network for Next Place Prediction
# Datasets
Experiment results on two Foursquare check-in datasets [[NYC,TKY](https://sites.google.com/site/yangdingqi/home/publication?authuser=0)] and one mobile phone dataset [CDRs]
# Requirements
* Python 3.6
* Pytorch 1.4
# Project Structure
* baselines
  * Markov.py
  * DeepMove.py [Paper](https://dl.acm.org/doi/abs/10.1145/3178876.3186058)
  * PLSPL.py [Paper](https://ieeexplore.ieee.org/abstract/document/9117156)
  * LSTPM.py [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5353)
* codes
  * main.py # setting parameters
  * train.py # train model
  * model.py # define models
  * utils.py # define tools
# Usage
* Train model
  #python main.py
# Result
top-1 accuracy | top-5 accuracy | top-10 accuracy 
 

