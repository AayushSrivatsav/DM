from collections import defaultdict

def eclat(transactions, min_support):
    """
    Implements the ECLAT algorithm to find frequent itemsets.

    Args:
        transactions: List of lists, where each inner list represents a transaction.
        min_support: Minimum support threshold (float).

    Returns:
        frequent_itemsets: Dictionary with itemsets as keys and their support as values.
    """
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

# Minimum support threshold
min_support = 0.5

# Run ECLAT
frequent_itemsets = eclat(transactions, min_support)

# Print Results
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")