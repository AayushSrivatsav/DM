#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:05:50 2025

@author: aayushsrivatsav
"""

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
