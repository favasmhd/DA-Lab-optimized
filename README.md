
# Apriori Algorithm – Simplified Code with Explanation

This is a shortened version of the Apriori algorithm implemented in Python using pandas. It finds frequent itemsets and generates association rules with confidence ≥ 0.5.

---

## 📄 CSV Format

Input file: `apriori.csv`  
Example:

```

TransactionID,Items
1,Milk,Bread
2,Bread,Butter
3,Milk,Bread,Butter

````

---

## 🧠 Code Explanation

**1. Imports and Setup**

```python
import pandas as pd
from itertools import combinations
````

* **pandas**: for reading the CSV.
* **combinations**: to generate all possible item combinations.

---

**2. Read Data and Initialize**

```python
df = pd.read_csv('apriori.csv', index_col='TransactionID')

counts = {}
freq_1 = {}
min_sup = 2
max_len = 0
```

* Reads the CSV file.
* **counts**: stores count of individual items.
* **freq\_1**: stores frequent 1-itemsets.
* **min\_sup**: minimum support threshold.
* **max\_len**: tracks the largest number of items in a transaction.

---

**3. Count Items and Find Frequent 1-Itemsets**

```python
for transaction in df['Items']:
    items = transaction.split(',')
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    if len(items) > max_len:
        max_len = len(items)

for it in counts:
    if counts[it] >= min_sup:
        freq_1[it] = counts[it]
```

* Splits each transaction into items and counts their frequency.
* Filters out infrequent items based on **min\_sup**.

---

**4. Generate Candidates (Size ≥ 2)**

```python
candidates = []
for size in range(2, max_len + 1):
    combos = list(combinations(freq_1.keys(), size))
    candidates.append(combos)
```

* For each size ≥ 2, generate all combinations of frequent items.

---

**5. Count Support for All Candidate Combos**

```python
support = {}
for combos in candidates:
    for combo in combos:
        support[combo] = 0
        for transaction in df['Items']:
            if all(i in transaction for i in combo):
                support[combo] += 1
```

* Count how many transactions contain each candidate itemset.

---

**6. Filter by Minimum Support**

```python
for combo in list(support.keys()):
    if support[combo] < min_sup:
        support.pop(combo)
```

* Remove itemsets that do not meet the minimum support.

---

**7. Generate and Print Association Rules**

```python
rules = []
for itemset, sup in support.items():
    length = len(itemset)
    if length < 2:
        continue

    for i in range(1, length):
        for left in combinations(itemset, i):
            right = tuple(set(itemset) - set(left))
            left_sup = counts[left[0]] if len(left) == 1 else support[left]
            conf = sup / left_sup

            if conf >= 0.5:
                rules.append({
                    "Rule": f"{left} -> {right}",
                    "Confidence": conf
                })

for r in rules:
    print(f"{r['Rule']}, Confidence: {r['Confidence']:.2f}")
```

* For each frequent itemset, generate all possible rules.
* Calculate confidence = **support(itemset) / support(left)**.
* Keep rules with confidence ≥ 0.5 and print them.

---

## ✅ Output Example

```
('Milk',) -> ('Bread',), Confidence: 0.75
```
---

# K-Means Clustering – Simplified Code with Explanation

This is a basic implementation of the K-Means clustering algorithm in Python using pandas and numpy.

---

## 📄 CSV Format

Input file: `kmeans.csv`  
Example:

```

no,height,weight
1,65,150
2,68,160
3,62,120
...

````

---

## 🧠 Code Explanation

**1. Imports and Reading Data**

```python
import pandas as pd
import random
import numpy as np

df = pd.read_csv("kmeans.csv", index_col='no')
````

* pandas: to read the CSV data.
* random: for selecting initial random centroids.
* numpy: for calculating means of clusters.

---

**2. Euclidean Distance Function**

```python
def euclidean(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
```

* Calculates Euclidean distance between two 2D points.

---

**3. Calculate New Centroids**

```python
def new_centroids(clusters):
    return [tuple(np.mean(cluster, axis=0)) for cluster in clusters]
```

* Finds the mean point (centroid) for each cluster.

---

**4. Initialization**

```python
k = int(input("Enter no of clusters: "))

c_index = []
while len(c_index) < k:
    x = random.randint(1, 10)
    if x not in c_index:
        c_index.append(x)

centroids = [tuple(df.loc[i]) for i in c_index]
points = list(zip(df['height'], df['weight']))
```

* Takes number of clusters (**k**) as input.
* Randomly selects **k** unique data points as initial centroids.
* Extracts all points as (height, weight) tuples.

---

**5. Clustering Loop**

```python
while True:
    clusters = [[] for _ in range(k)]
    for point in points:
        distances = [euclidean(point, centroid) for centroid in centroids]
        clusters[distances.index(min(distances))].append(point)
    new_centroids_list = new_centroids(clusters)
    if new_centroids_list == centroids:
        break
    centroids = new_centroids_list
```

* Assigns each point to the nearest centroid’s cluster.
* Calculates new centroids of clusters.
* Stops when centroids no longer change.

---

**6. Output**

```python
print(clusters)
```

* Prints the final clusters as lists of points.

---

# Naive Bayes Classifier – Simplified Code with Explanation

This is a basic implementation of the Naive Bayes classifier using pandas for a categorical dataset.

---

## 📄 CSV Format

Input file: `naive.csv`  
Example columns: `age`, `income`, `student`, `credit`, `buys` (target variable)

---

## 🧠 Code Explanation

**1. Import and Read Data**

```python
import pandas as pd

df = pd.read_csv("naive.csv")
````

* pandas: to read the CSV dataset.

---

**2. Calculate Frequency Counts**

```python
age = df.groupby(['age','buys']).size()
income = df.groupby(['income','buys']).size()
student = df.groupby(['student','buys']).size()
credit = df.groupby(['credit','buys']).size()

collection = [age, income, student, credit]
```

* Groups data by each feature and target (`buys`) to count occurrences.
* Stores frequency counts for later probability calculations.

---

**3. Calculate Prior Probabilities**

```python
total_yes = df['buys'].value_counts()['yes']
total_no = df['buys'].value_counts()['no']
total = len(df)

prior_yes = total_yes / total
prior_no = total_no / total
```

* Counts total "yes" and "no" labels.
* Computes prior probabilities P(yes) and P(no).

---

**4. Define Input and Classification Function**

```python
input_data = {'age':'youth', 'income':'medium', 'student':'yes', 'credit':'fair'}

def naive_classify(data):
    yes = prior_yes
    no = prior_no
    for i, j in zip(data.values(), collection):    
        yes *= j.get((i, 'yes'), 0) / total_yes
        no *= j.get((i, 'no'), 0) / total_no
    return float(yes), float(no)
```

* Takes input features to classify.
* Calculates likelihood P(features|class) and multiplies by prior (Naive Bayes assumption).

---

**5. Run Classification and Output**

```python
yes, no = naive_classify(input_data)
print(input_data)
print(f"P(Yes|x): {yes}")
print(f"P(No|x): {no}")

if yes > no:
    print("Yes has more probability")
else:
    print("No has more probability")
```

* Prints posterior probabilities for classes "yes" and "no".
* Outputs which class has higher probability.

---

Let me know if you want me to prepare explanations for any other codes!
