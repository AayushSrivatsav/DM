#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:59:29 2025

@author: aayushsrivatsav
"""

from itertools import combinations

def get_support(itemset, transactions):
    """Calculate the support of an itemset."""
    return sum(1 for transaction in transactions if itemset.issubset(transaction)) / len(transactions)

def generate_candidates(prev_frequent, k):
    """Generate candidate itemsets of length k from the previous frequent itemsets."""
    candidates = set()
    prev_frequent_list = list(prev_frequent)
    
    for i in range(len(prev_frequent_list)):
        for j in range(i + 1, len(prev_frequent_list)):
            union_set = prev_frequent_list[i] | prev_frequent_list[j]
            if len(union_set) == k:
                candidates.add(union_set)
    return candidates

def apriori(transactions, min_support):
    """Apriori algorithm to find frequent itemsets."""
    frequent_itemsets = []
    single_items = {frozenset([item]) for transaction in transactions for item in transaction}
    
    current_frequent = {item for item in single_items if get_support(item, transactions) >= min_support}
    k = 2
    
    while current_frequent:
        frequent_itemsets.extend(current_frequent)
        candidates = generate_candidates(current_frequent, k)
        current_frequent = {itemset for itemset in candidates if get_support(itemset, transactions) >= min_support}
        k += 1
    
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    """Generate strong association rules from frequent itemsets."""
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:  # Only generate rules for sets with more than one item
            for i in range(1, len(itemset)):
                for subset in combinations(itemset, i):
                    antecedent = frozenset(subset)
                    consequent = itemset - antecedent
                    support_antecedent = get_support(antecedent, transactions)
                    support_itemset = get_support(itemset, transactions)
                    confidence = support_itemset / support_antecedent if support_antecedent > 0 else 0

                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))

    return rules

# Transactions dataset
transactions = [
    {'Milk', 'Bread', 'Eggs'},
    {'Bread', 'Butter', 'Eggs'},
    {'Milk', 'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Butter', 'Cheese', 'Eggs'},
    {'Bread', 'Butter', 'Cheese'},
    {'Milk', 'Bread', 'Eggs'}
]

# Parameters
min_support = 0.3
min_confidence = 0.7  # Example: minimum confidence of 70%

# Run Apriori Algorithm
frequent_itemsets = apriori(transactions, min_support)
print("Frequent Itemsets:", frequent_itemsets)

# Generate and print strong association rules
strong_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
print("\nStrong Association Rules:")
for rule in strong_rules:
    print(f"{set(rule[0])} => {set(rule[1])} (Confidence: {rule[2]:.2f})")