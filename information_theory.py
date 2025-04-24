import math
from collections import Counter

def shannon_entropy(class_probs):
    return -sum(p * math.log2(p) for p in class_probs if p > 0)

# For a binary classification: [0.5, 0.5]
#shannon_entropy([0.5, 0.5])  # Output: 1.0 (maximum uncertainty)


def cross_entropy(p_true, q_pred):
    return -sum(p * math.log2(q) for p, q in zip(p_true, q_pred) if p > 0 and q > 0)

#p_true = [1, 0]       # Actual class is 0
#q_pred = [0.9, 0.1]   # Model's prediction
#cross_entropy(p_true, q_pred)  # Output: ~0.152


def entropy_from_labels(labels):
    total = len(labels)
    counts = Counter(labels)
    probs = [count / total for count in counts.values()]
    return shannon_entropy(probs)

def information_gain(parent_labels, split_subsets):
    parent_entropy = entropy_from_labels(parent_labels)
    total = len(parent_labels)
    
    weighted_entropy = 0
    for subset in split_subsets:
        weighted_entropy += (len(subset) / total) * entropy_from_labels(subset)
    
    return parent_entropy - weighted_entropy

#parent = ['yes', 'yes', 'no', 'no', 'yes']
#split = [
 #   ['yes', 'yes'],   # left branch
 #   ['no', 'no', 'yes']  # right branch
#]
#information_gain(parent, split)  # Output: IG value

def kl_divergence(p, q):
    """KL Divergence D_KL(P || Q)"""
    return sum(
        pi * math.log2(pi / qi)
        for pi, qi in zip(p, q)
        if pi > 0 and qi > 0
    )

# True distribution P
#p = [0.6, 0.4]

# Predicted distribution Q
#q = [0.5, 0.5]

#kl_divergence(p, q)  # Output: ~0.029 (measured in bits)

def js_divergence(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

#p = [0.6, 0.4]
#q = [0.5, 0.5]

#js_divergence(p, q)  # Output: ~0.0075

def js_distance(p, q):
    return math.sqrt(js_divergence(p, q))


