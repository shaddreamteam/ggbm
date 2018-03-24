import numpy as np

class Subleaf:
    def __init__(self):
        self.G = 0
        self.H = 0
        self.count = 0
        self.gain = 0

    def add(self, g, h):
        self.G += g
        self.H += h
        self.count += 1
        
    def summarise(self, lambda_val):
        if self.count == 0:
            self.weight = 0
            self.gain = 0
        else:
            self.weight = -self.G / (self.H + self.count * lambda_val)
            self.gain = -self.G ** 2 / (self.H + self.count * lambda_val) / 2
        return self.gain, self.count

    def copy(self):
        copy = Subleaf()
        copy.G = self.G
        copy.H = self.H
        copy.count = self.count
        copy.gain = self.gain
        return copy
        
class Leaf:
    def __init__(self, entries, weight, path):
        self.entries = entries
        self.path = path
        self.weight = weight
        self.final = False
        self.left, self.right = Subleaf(), Subleaf()
            
    def get_split_weigths_gain(self, feature, bin_number, lambda_val):
        for entry in self.entries:
            if entry.x[feature] <= bin_number:
                self.left.add(entry.g, entry.h)
            else:
                self.right.add(entry.g, entry.h)
        gain_left, count_left = self.left.summarise(lambda_val)
        gain_right, count_right = self.right.summarise(lambda_val)
        return gain_left + gain_right, min(count_left, count_right)

    def remember_split(self, feature, bin_num):
        self.saved_left_weight = self.left.weight
        self.saved_right_weight = self.right.weight
        self.saved_feature = feature
        self.saved_bin = bin_num
        
    def get_new_leafs(self):
        left_entries = [e for e in self.entries if e.x[self.saved_feature] <= 
                        self.saved_bin]
        left_leaf = Leaf(left_entries, self.saved_left_weight, 
                         [x for x in self.path].append(0))
        
        right_entries = [e for e in self.entries if e.x[self.saved_feature] > 
                        self.saved_bin]
        right_leaf = Leaf(right_entries, self.saved_right_weight,
                          [x for x in self.path].append(1))
        return left_leaf, right_leaf

class Tree():
    def __init__(self, depth, loss, entries, tresholds, lambda_val, min_leaf_count, gamma):
        self.depth = depth
        self.splits = []
        self.leafs = [Leaf(entries, 0, [])]
        self.tresholds = tresholds
        self.path_leaf_dict = dict()
        
    
    def construct(self):
        depth_counter = 0
        no_gain_flag = False
        best_gain = 0
        too_small_splits = set()
        while depth_counter < self.depth and not no_gain_flag:
            previous_gain = best_gain
            best_gain = 0
            for feature in len(self.tresholds):
                for bin_num in len(self.tresholds[feature_num]):
                    if (feature, bin_num) not in self.splits and\
                       (feature, bin_num) not in too_small_splits:
                        bin_gain = 0
                        for leaf in self.leafs:
                            leaf_gain, min_split_count = \
                                leaf.get_split_weigths_gain(feature, bin_num, self.lambda_val)
                            if min_split_count < self.min_leaf_count:
                                too_small_splits.add((feature, bin_num))
                                break
                        if bin_gain < best_gain and (min_split_count >= self.min_leaf_count):
                            best_gain = bin_gain
                            best_split = (feature, bin_num)
                            for leaf in self.leafs:
                                leaf.remember_split()
            if best_gain + self.gamma >= previous_gain:
                no_gain_flag = True
                break
            else:
                self.splits.append(best_split)
                new_leafs = []
                for leaf in self.leafs:
                    left, right = leaf.get_new_leafs()
                    new_leafs.append(left)
                    new_leafs.append(right)
                self.leafs = new_leafs
                depth_counter += 1
                
        for leaf in self.leafs:
            self.path_leaf_dict[leaf.path] = leaf        
    def predict(entry):
        path = []
        for split in self.splits:
            path.append(entry.x[split[0]] > split[1])
        return self.path_leaf_dict[path].weight
