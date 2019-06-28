
### Questions
* How is the train test split supposed to work?
* Are you supposed to ignore the target completely in a train test split?
* when calculating the mean & standard deviation  (using .groupby and .agg), are we really supposed to use the entire dataset instead of the training dataset

### Objectives
YWBAT
* Describe how Naive Bayes is used to classify
* Describe BOW in document classification

### Outline
* Questions
* Going over the last 2 labs from section 23
* Discuss the purpose of NB in ML and how it's used and when it's used

### Naive Bayes
* What the purpose of NB?
    * To classify data into classes/groups

* How does it classify?
    * Checks the probability for each class P(class | row)
    * Bayes Theorem is (P(class | row) = P(class) * P(row | class)) / P(row)
    * Relies on the assumption that distributions are identical
        * Probabilities can be set the same
        * Need mu and std which comes from sample with that class
        * what is the sample though?
        * sample = training set
    * In Naive Bayes why are these probabilities so low?
        * Not really why, but, Because we dropped the denominator
        * It's because we're multiplying small probabilities


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
```

### Assessing Classifier
- Train and Test accuracies should be equal (ideally)


**How do you avoid training a model that is overfitted because of a bad train/test split?**
- KFolds cross Validation

Parametric Models
- Linear Regression
    - What are the parameters? 
        * Beta coefficients 

- Naive Bayes
    - Where 'theta' is the parameter

When do we use NB?
- NLP
    - Text classification
    - Document classification
    - Maybe sentiment - unsolved problem
    
    
What is NLP?
- Field of Data Science that analyzes and measures various aspects of natural language
- Word Counts
- Word Importance
- Word Density (Word Frequency / Words in Doc)
- tfidf - looks at word importance over several documents
    - common words -> 0
    - rare/topical words don't go to zero
 - Word Vectorization
 - Document Vectorization
 - Sentence Vectorization
 - Classify words, docs, sentences based on vectorizations

### Assessment
* More about NLP - Mapping sentences to vectors using BOW
* Computing probabilities based on training distributions
* General information about NB and NLP
