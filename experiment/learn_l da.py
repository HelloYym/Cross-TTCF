import lda
import numpy as np

X = lda.datasets.load_reuters()
titles = lda.datasets.load_reuters_titles()
X_train = X[300:]
X_test = X[:300]
titles_test = titles[:10]
model = lda.LDA(n_topics=20, n_iter=100, random_state=1)
model.fit(X_train)
doc_topic_test = model.transform(X_test)
for title, topics in zip(titles_test, doc_topic_test):
    print("{} (top topic: {})".format(title, topics.argmax()))
