# Fake News Classification using Traditional Machine Learning Models

### Problem Statement  

Fake news detection is still in the early age of development, and there are still many challenging issues that need further investigations. It is necessary to discuss potential research directions that can improve fake news detection and mitigation capabilities. Binary classification is done by using different machine learning algorithms.

### Dataset 

* True.csv: A full dataset containing true news
  * title: the title of a news article
  * text: the text of the article; could be incomplete
  * subject: the subject of the article 
  * date: date of the news
  * target: label of the article - true or fake

* Fake.csv: A full dataset with all the same attributes at True.csv but containing fake news

### File Structure
The file structure is the following
```
.
|
+-- requirements.txt
+-- data
|   +-- True.csv
|   +-- Fake.csv
+-- modelling
|   +-- [ML]fake-news-classification.ipynb
+-- demo
|   +-- app.py
|   +-- svm_model.pkl
```

### Install requirements
```{r, engine='bash', count_lines}
$ pip install -r requirements.txt
```

### Train BHDD with Basic ConvNet Architecture with Dropout

```{r, engine='bash', count_lines}
$ runipy [ML]fake-news-classification.ipynb
```
### For Demo Only

```{r, engine='bash', count_lines}
$ streamlit run app.py
```

- Confusion matrices after training with Different Classifiers
![Confusion matrix Images](Image/matrix.png)


### Comparing Accuracies of Machine Learning Models

| Model                     | Accuracy     |
|:-------------------------:|:------------:|
| Naive Bayes               | 95.11%       |
| Logistic Regression       | 98.98%       |
| Decision Tree             | 99.69%       |
| Random Forest             | 99.04%       |
| Linear SVM                | 99.65%       |
| SVM with RBF kernel       | 99.52%       |

- Decision Tree classifier seems to be the best fit on the dataset. So we shall use decision tree classifier on streamlit. 

### References

  * [Fake News Identification - Stanford CS229](http://cs229.stanford.edu/proj2017/final-reports/5244348.pdf)
  * [Machine Learning for Detection of Fake News](https://dspace.mit.edu/bitstream/handle/1721.1/119727/1078649610-MIT.pdf)
  * [Fake News Classification](https://github.com/SauravMaheshkar/Fake-News-Classification)
  * [Datasets from Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
