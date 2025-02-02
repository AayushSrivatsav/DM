
def apriori(minsup,transaction):
    def calculate_support(itemset):
        count=0
        for i in transaction:
            if itemset.issubset(i):
                count+=1
        return count
    itemsets = {frozenset([item]) for t in transaction for item in t}
    print(itemsets)
    freqitemsets={}
    k=1
    while itemsets:
        itemsets_support = {itemset:calculate_support(itemset) for itemset in itemsets}
        for i in itemsets_support:
            if itemsets_support[i]>=minsup:
                freqitemsets[i] = itemsets_support[i]
        k+=1
        itemsets = set(frozenset(i.union(j)) for i in freqitemsets for j in freqitemsets if len(i.union(j))==k)
    return freqitemsets

transactions = [
    {'milk', 'bread', 'butter'},
    {'milk', 'bread'},
    {'milk', 'butter'},
    {'bread', 'butter'},
    {'milk', 'bread', 'butter', 'eggs'},
    {'bread', 'butter', 'eggs'},
]
min_support = 2
frequent_itemsets = apriori(min_support,transactions)
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")
    
    
#rules
#Association Rule Mining
def generate_rules(transaction, min_s, min_c):
    def calculate_support(itemset):
        count =0
        for i in transaction:
            if itemset.issubset(i):
                count += 1
        return count/len(transaction)
    def frequent_set():
        itemset = {frozenset([item]) for t in transaction for item in t}
        freqset = {}
        k = 1

        while itemset:
            itemset_support = {item: calculate_support(item) for item in itemset}
            itemset = {item for item in itemset_support if itemset_support[item] >= min_s}
            freqset.update(itemset_support)

            candidates = {frozenset(a.union(b)) for a, b in combinations(itemset, 2) if len(a.union(b)) == k + 1}
            itemset = candidates
            k += 1

        return freqset

    def generate_association_rules(freq, min_c):
        rules = []
        
        for itemset in freq.keys():
            if len(itemset) < 2:
                continue
            
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in freq:
                        confidence = freq[itemset] / freq[antecedent]
                        if confidence >= min_c:
                            support = freq[itemset]
                            rules.append((antecedent, consequent, support, confidence))

        return rules

    frequent_itemsets = frequent_set()
    rules = generate_association_rules(frequent_itemsets, min_c)
    
    return frequent_itemsets, rules

transactions = [
    {"Milk", "Bread", "Butter"},
    {"Milk", "Bread"},
    {"Bread", "Butter"},
    {"Milk", "Butter"},
]

min_support = 0.5
min_confidence = 0.6

frequent_itemsets, rules = generate_rules(transactions, min_support, min_confidence)

print("\nFrequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)} : {support:.2f}")

print("\nAssociation Rules:")
for antecedent, consequent, support, confidence in rules:
    print(f"{set(antecedent)} → {set(consequent)} (Support: {support:.2f}, Confidence: {confidence:.2f})")

