# Spam Classification Using Machine Learning and GPT

This project evaluates classical machine learning models and a modern large language model (GPT) for the task of spam detection. The experiments compare performance, generalization ability, and computational cost across two datasets.

---

## 1. Datasets

Two public spam datasets were used.  
Each entry contains:

- text: the message content  
- label: "spam" or "ham"

The goal is to classify each message as spam or not spam.

Dataset diversity helps evaluate the robustness of all models.

---

## 2. Methodology

The project follows two parallel pipelines:

1. Classical machine learning models trained using TF-IDF vectorization.  
2. GPT evaluated using zero-shot prompting, without training.  

All classical models were evaluated using stratified 5-fold cross-validation.

Metrics computed:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Execution time  

GPT used the same test splits for fairness.

---

## 3. Text Vectorization (TF-IDF)

All models (except GPT) receive messages transformed into TF-IDF features.

TF-IDF formula:

TF-IDF(word, doc) = TF(word, doc) × log(N / df(word))


Where:

- TF(word, doc): frequency of a word in the document  
- N: total number of documents  
- df(word): number of documents containing the word  

Why TF-IDF is used:

- Converts text into numerical vectors  
- Gives higher weight to informative words  
- Works extremely well with traditional ML models  

Configuration:

max_features=3000
stop_words='english'
ngram_range=(1, 2)
min_df=2
max_df=0.95


This provides a balanced and informative feature space.

---

# 4. Classical Machine Learning Models

Each model below includes:

- A simple explanation  
- The mathematical intuition  
- Why we used it in the project  

---

## 4.1 Logistic Regression

Predicts the probability of a message being spam.

Probability function:

P(spam | x) = 1 / (1 + e^(-(wᵀ x + b)))


Loss minimized:

CrossEntropy = -[y log(p) + (1 - y) log(1 - p)]


Why use Logistic Regression:

- Strong baseline for text classification  
- Very robust in high-dimensional TF-IDF spaces  
- Provides well-calibrated probabilities  
- Fast and interpretable  

---

## 4.2 Multinomial Naive Bayes

A probabilistic model assuming word independence.

Formula:

P(spam | doc) ∝ P(spam) × Π P(wordᵢ | spam)


Why use Naive Bayes:

- Extremely fast  
- Excellent for large sparse text datasets  
- Often surprisingly strong for spam filtering  
- Provides a lower bound baseline  

In the experiments, it was the fastest model by far.

---

## 4.3 Support Vector Machine (Linear SVM)

Finds the best hyperplane to separate spam and ham.

Hard-margin formulation:

Minimize: ||w||²
Subject to: yᵢ (wᵀ xᵢ + b) ≥ 1


Soft-margin version:

Minimize: (1/2)||w||² + C Σ ξᵢ
Subject to: yᵢ (wᵀ xᵢ + b) ≥ 1 - ξᵢ
ξᵢ ≥ 0


Why use SVM:

- Excellent performance in high-dimensional sparse spaces  
- One of the best models for text classification  
- Handles noisy data well  
- Frequently achieves top F1 scores in spam tasks  

---

## 4.4 Decision Tree

A rule-based model that splits features into regions.

Why use Decision Trees:

- Interpretable  
- Fast to train  
- Serves as a weak baseline for comparison  
- Allows understanding of model limitations in sparse, high-dimensional text  

Decision Trees typically perform poorly on text, but including them shows how more advanced methods improve results.

---

## 4.5 Random Forest

An ensemble of multiple decision trees.

Why use Random Forest:

- More robust than a single tree  
- Reduces overfitting  
- Good at handling nonlinear patterns  
- Well-established for tabular data  

In our experiments, Random Forest produced strong F1 scores.

---

## 4.6 Gradient Boosting

A sequential ensemble where each new tree corrects mistakes from prior ones.

General update:

Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)


Why use Gradient Boosting:

- High predictive power  
- Captures nonlinearities better than linear models  
- Widely used in practical machine learning tasks  

It was the slowest classical model due to sequential operations.

---

## 4.7 Ensemble Methods (Voting and Weighted)

### Soft Voting

P(spam) = (1/N) Σ Pᵢ(spam)


### Weighted Voting


Why use Ensembles:

- Combine strengths of multiple models  
- Reduce variance  
- Improve stability and generalization  

In Dataset 2, the Weighted Ensemble was the best overall model.

---

# 5. Evaluation Method: Stratified K-Fold

K-fold ensures that every model is tested fairly.

Process:

1. Split dataset into 5 folds preserving spam/ham ratio  
2. Train on 4 folds, test on the remaining fold  
3. Repeat 5 times  
4. Compute average metrics  

Why use K-Fold:

- Reduces variance in evaluation  
- Prevents train-test split bias  
- Allows fair comparison across models  

GPT also uses the same test splits (but without training).

---

# 6. GPT Zero-Shot Classification

GPT was evaluated **without any training** using a prompt such as:

You are a spam detection expert.
Classify the following message as spam or ham.
Respond with one word.


Why use GPT:

- Understands semantics beyond keyword frequency  
- Can generalize without training  
- Useful for analyzing meaning and context  
- Demonstrates modern NLP capability compared to traditional models  

Evaluation:

- GPT only sees test data  
- Predictions compared to true labels  
- Same folds as classical models  

Key results:

- Very high recall (~0.978)  
- Lower precision (~0.73)  
- Competitive F1 (~0.84)  

GPT tends to mark borderline messages as spam, favoring safety.

---

# 7. Summary of Performance

Classical model insights:

- SVM, Logistic Regression, Random Forest, and Weighted Ensemble performed best  
- Naive Bayes was the fastest  
- Ensembles produced the most consistent predictions  

GPT insights:

- Excellent semantic understanding  
- Highest recall among all models  
- Lower precision due to more false positives  
- Zero training required  

---

# 8. Conclusion

Traditional machine learning models remain highly effective for spam detection when combined with TF-IDF.  
Ensembles deliver the most stable performance and highest accuracy.  
GPT, despite being used in zero-shot mode, achieved strong results by capturing semantic patterns instead of surface-level word frequencies.

Future improvements may include:

- Few-shot prompting  
- Hybrid classical + GPT ensembles  
- Fine-tuning on spam-specific datasets  
- Embedding-based classifiers using GPT or other transformer models
