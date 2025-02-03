#apriori
from itertools import cominations

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




#Hashh
import itertools

def hash_apriori(minsup, transactions):
    def calculate_support(itemset):
        count = sum(1 for t in transactions if itemset.issubset(t))
        return count / len(transactions)

    itemsets = {frozenset([item]) for t in transactions for item in t}
    num_buckets = 7  # User-defined
    buckets = {i: [] for i in range(num_buckets)}
    freqitemsets = {}

    # Assign each item a unique numeric value for hashing
    values = {item: idx + 1 for idx, item in enumerate(set().union(*transactions))}

    def custom_hash(itemset, num_buckets):
        hash_value = 1
        for item in itemset:
            hash_value *= values.get(item, 1)  
        return hash_value % num_buckets

    k = 1  
    while itemsets:
        # Compute support for all itemsets
        itemset_support = {itemset: calculate_support(itemset) for itemset in itemsets}

        # Prune non-frequent itemsets
        itemsets = {itemset for itemset, support in itemset_support.items() if support >= minsup}
        freqitemsets.update({itemset: itemset_support[itemset] for itemset in itemsets})

        if not itemsets:
            break  

        # Candidate generation
        itemsets = {
            frozenset(i.union(j))
            for i in itemsets
            for j in itemsets
            if len(i.union(j)) == k + 1 and all(frozenset(sub) in freqitemsets for sub in itertools.combinations(i.union(j), k))
        }

        # Hashing step
        for itemset in itemsets:
            hash_value = custom_hash(itemset, num_buckets)
            buckets[hash_value].append(itemset)

        k += 1  

    return freqitemsets

transactions = [
    {'milk', 'bread', 'butter'},
    {'milk', 'bread'},
    {'milk', 'butter'},
    {'bread', 'butter'},
    {'milk', 'bread', 'butter', 'eggs'},
    {'bread', 'butter', 'eggs'},
]

# Minimum support threshold
min_support = 0.3

# Run Apriori with Hash Table Improvement
frequent_itemsets = hash_apriori(min_support, transactions)

# Print Results
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")



#eclatt
from collections import defaultdict

def eclat(transactions, min_support):

    # Convert transactions to vertical format (item-TIDsets)
    vertical_db = defaultdict(set)
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            vertical_db[frozenset([item])].add(tid)

    # Minimum support count
    min_support_count = min_support * len(transactions)

    # Recursive function to mine frequent itemsets
    def recursive_eclat(prefix, tidsets):
        frequent_itemsets = {}
        for i, (itemset, tidset) in enumerate(tidsets.items()):
            # Compute support
            support = len(tidset)
            if support >= min_support_count:
                new_itemset = prefix.union(itemset)
                frequent_itemsets[frozenset(new_itemset)] = support / len(transactions)
                
                # Recursive step
                remaining_tidsets = {other_itemset: tidset & other_tidset
                                     for j, (other_itemset, other_tidset) in enumerate(tidsets.items())
                                     if j > i}
                frequent_itemsets.update(recursive_eclat(new_itemset, remaining_tidsets))
        return frequent_itemsets

    # Start the recursive mining process
    return recursive_eclat(frozenset(), vertical_db)


# Sample dataset
transactions = [
    {'milk', 'bread', 'butter'},
    {'milk', 'bread'},
    {'milk', 'butter'},
    {'bread', 'butter'},
    {'milk', 'bread', 'butter', 'eggs'},
    {'bread', 'butter', 'eggs'},
]
for a,b in enumerate(transactions):
    print(a,b)

# Minimum support threshold
min_support = 0.5

# Run ECLAT
frequent_itemsets = eclat(transactions, min_support)

# Print Results
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")
