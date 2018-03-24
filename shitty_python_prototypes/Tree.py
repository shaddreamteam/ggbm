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
        return self.gain

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
            
    def get_split_weigths_gain(self, feature, bin_number, lambda_val):
        self.left, self.right = Subleaf(), Subleaf()
        for entry in self.entries:
            if entry.x[feature] <= bin_number:
                self.left.add(entry.g, entry.h)
            else:
                self.right.add(entry.g, entry.h)
        gain = 0
        gain += self.left.summarise(lambda_val)
        gain += self.right.summarise(lambda_val)
        return gain

    def remember_split(self, feature, bin_num):
        self.saved_left_weight = self.left.weight
        self.saved_right_weight = self.right.weight
        self.saved_feature = feature
        self.saved_bin = bin_num
        self.saved_left_count = self.left.count
        self.saved_right_count = self.right.count
        
    def get_new_leafs(self):
        if self.saved_left_weight == 0 and self.saved_right_weight == 0:
            return [self]
        
        new_leafs = []
        if self.saved_left_weight != 0:
            left_path = [x for x in self.path]
            left_path.append(0)
            left_entries = [e for e in self.entries if e.x[self.saved_feature] <= 
                            self.saved_bin]
            new_leafs.append(Leaf(left_entries, self.saved_left_weight, left_path))
            
        if self.saved_right_count != 0:
            right_path = [x for x in self.path]
            right_path.append(1)
            right_entries = [e for e in self.entries if e.x[self.saved_feature] > 
                            self.saved_bin]
            new_leafs.append(Leaf(right_entries, self.saved_right_weight, right_path))
        return new_leafs

class Tree():
    def __init__(self, depth, loss, entries, tresholds, lambda_val, min_leaf_count, gamma):
        self.depth = depth
        self.loss = loss
        self.leafs = [Leaf(entries, 0, [])]
        self.tresholds = tresholds
        self.lambda_val = lambda_val
        self.min_leaf_count = min_leaf_count
        self.gamma = gamma
        self.splits = []
        self.path_leaf_dict = dict()
        
    
    def construct(self):
        depth_counter = 0
        no_gain_flag = False
        best_gain = 0
        while depth_counter < self.depth and not no_gain_flag:
            previous_gain = best_gain
            best_gain = 0
            for feature in range(len(self.tresholds)):
                for bin_num in range(len(self.tresholds[feature])):
                    if (feature, bin_num) not in self.splits:
                        bin_gain = 0
                        for leaf in self.leafs:
                            bin_gain += leaf.get_split_weigths_gain(feature, bin_num, self.lambda_val)
                        if bin_gain < best_gain:
                            best_gain = bin_gain
                            best_split = (feature, bin_num)
                            for leaf in self.leafs:
                                leaf.remember_split(feature, bin_num)
            if best_gain + self.gamma >= previous_gain:
                no_gain_flag = True
                break
            else:
                self.splits.append(best_split)
                new_leafs = []
                for leaf in self.leafs:
                    for new_leaf in leaf.get_new_leafs():
                        new_leafs.append(new_leaf)
                self.leafs = new_leafs
                depth_counter += 1
                
        for leaf in self.leafs:
            self.path_leaf_dict[str(leaf.path)] = leaf

    def predict(self, entry):
        path = []
        for split in self.splits:
            path.append(int(entry.x[split[0]] > split[1]))
        return self.path_leaf_dict[str(path)].weight
