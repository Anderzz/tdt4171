import numpy as np
import pandas as pd
from graphviz import Digraph
import random
import string

random.seed(10)
target = '1.5'

class Node:
    """
    Class Node
    """
    def __init__(self, value):
        self.children = []
        self.value = value
        self.id = get_id()
            


def B(q):
    #p. 1222 in the book
    #returns the entropy
    return 0 if q == 1 or q == 0 else -(q*np.log2(q) + (1-q)*np.log2(1-q))

def remainder(tar, A, p, n, examples):
    #p 1223 in the book
    #returns the expected remaining entropy after testing A
    distinct_vals = examples[A].unique()
    splits = []
    for val in distinct_vals:
        splits.append(examples[examples[A] == val])

    sum = 0
    for split in splits:
        try:
            pk = split[tar].value_counts()[1]
        except:
            pk = 0
        try:
            nk = split[tar].value_counts()[2]
        except:
            nk = 0
    
        sum += ((pk + nk)/(p + n)) * B(pk/(pk + nk))
    return sum

def importance(tar, A, examples, rand=False):
    #p. 1223 in the book
    #the information gain from the attribute test on A is the expected reduction in entropy
    if rand: return random.uniform(0,1) 
    #positive
    try:
        p = examples[tar].value_counts()[1]
    except:
        p = 0
    #negative
    try:
        n = examples[tar].value_counts()[2]
    except:
        n = 0

    return B(p/(p+n)) - remainder(tar, A, p, n, examples)

def get_id():
    #make a pseudorandom 10 character id
    return (''.join(random.choices(string.ascii_lowercase, k=10)))

def plurality_values(e, tree=None):
    #return the most common value
    value = e[target].mode()[0]
    id = get_id()
    tree.node(id, label = str(value))
    return [id, str(value)]

def same_classification(e: pd.Series, tree: Digraph):
    #return the classification
    classificaion = e.unique()[0]
    id = get_id()
    tree.node(id, label = str(classificaion))
    return [id, str(classificaion)]

def traverse(dict, row):
    #traverse the tree (dictionary), used for testing the model
    if dict == '1' or dict == '2': # stopping condition
        return dict
    for key in dict:
        next = row.loc[key]
    return traverse(dict[key][next], row) #recurse

def learn_decision_tree(examples, attributes, tree=None, parent_examples=(), rand = False):

    if examples.empty:
        return plurality_values(parent_examples,tree)

    elif len(examples[target].unique()) == 1:
        return same_classification(examples[target], tree)

    elif attributes.empty:
        return plurality_values(examples,tree)
    
    #find the most important attribute
    else:
        model = {} #store the model
        gain = {} #information gain
        for col in attributes:
            gain[col] = importance(target, col, examples, rand)
        
        max_list = sorted(gain.items(), key=lambda x: x[1], reverse=True)
        max_list = list(filter(lambda x: x[0] != '1.5', max_list))
        A = max_list[0][0]
        id = get_id()
        tree.node(id, label=A)
        subtree_dict = {} #used to the subtree under each v
        for v in examples[A].unique():
            exs = examples[examples[A]==v]
            new_attrs = attributes.copy().drop(A)
            subtree = learn_decision_tree(exs, new_attrs, tree, examples, rand=rand)
            tree.edge(id, subtree[0], label=str(v))
            subtree_dict[v] = subtree[1] #add the subtree under v
        
        model[A] = subtree_dict
        return [id, model]


def main():

    # load the data and rename clunky attributes
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train.rename(columns={'1': 'a1', '2': 'a2'}, inplace=True)
    test.rename(columns={'1': 'a1', '2': 'a2'}, inplace=True)

    #initialize the tree
    tree = Digraph(name="Decision Tree", filename="dtl.dot")

    #learn the model
    res = learn_decision_tree(train, train.columns, tree, rand = False)[1]

    #draw the tree
    tree.render(view=True)
    
    ########### test the model ###########
    right = 0 
    wrong = 0
    for _, row in test.iterrows():
        prediction = traverse(res, row)
        if prediction == str(row.loc['1.4']): #did we get it right?
            right += 1
        else:
            wrong += 1
    print(f"\nModel predicted {right} correct and {wrong} wrong. \nAccuracy = {round(right/(right+wrong),3)}")
    ###########  end test ###########


if __name__ == "__main__":
    main()
