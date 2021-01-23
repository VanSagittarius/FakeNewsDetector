# FakeNewsDetector
A type of yellow journalism, fake news encapsulates pieces of news that may be hoaxes and is generally spread through social media and other online media.

In order to properly mark news (if they are FALSE or REAL) we can use **TfidfVectorizer**, whcih can covert collection of raw documents into a matrix of TF-IDF features:
    - **TF (Term Frequency)** – The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms
    - **IDF (Inverse Document Frequency** – Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus

**PassiveAggressiveClassifier** is also helpful in this field – algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

Project:
1. Using `sklearn`, we build `TfidVectorizer` on our data set
2. Initialize a `PassiveAggressiveClassifier` and fit the model
3. Printing the accuracy score and confusion matrix to estiminated model result
