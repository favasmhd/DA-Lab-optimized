import pandas as pd
from itertools import combinations

df = pd.read_csv('apriori.csv', index_col='TransactionID')

counts = {}
freq_1 = {}
min_sup = 2
max_len = 0

for transaction in df['Items']:
    items = transaction.split(',')
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    if len(items) > max_len:
        max_len = len(items)

for it in counts:
    if counts[it] >= min_sup:
        freq_1[it] = counts[it]

candidates = []
for size in range(2, max_len + 1):
    combos = list(combinations(freq_1.keys(), size))
    candidates.append(combos)

support = {}

for combos in candidates:
    for combo in combos:
        support[combo] = 0
        for transaction in df['Items']:
            if all(i in transaction for i in combo):
                support[combo] += 1

for combo in list(support.keys()):
    if support[combo] < min_sup:
        support.pop(combo)

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
