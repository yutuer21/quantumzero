
#%matplotlib inline

from __future__ import division
from node import Node
from result import Result
import collections
import random
import numpy as np
import sys
import math
import ast
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt



class Tree:
    def __init__(self,data, T,get_reward, positions_order="reverse", max_flag=True, expand_children=1,
                 space=None, candidate_pool_size=None, no_positions=None, atom_types=None, atom_const=None, play_out=1, play_out_selection="best",ucb="mean"):
        print("mcts")
        self.data=data
        self.T=T
        if space is None:
            self.space=None
            if (no_positions is None) or (atom_types is None):
                sys.exit("no_positions and atom_types should not be None")
            else:
                self.no_positions = no_positions
                self.atom_types = atom_types
                self.atom_const = atom_const
      
                self.candidate_pool_size = candidate_pool_size
        else:
            self.space = space.copy()
            self.one_hot_space=self.one_hot_encode(self.space)
            self.no_positions = space.shape[1]
            self.atom_types = np.unique(space)     ####unique 是指只记录其间出现的可能性

        if positions_order == "direct":
            self.positions_order = list(range(self.no_positions))
        elif positions_order == "reverse":
            self.positions_order = list(range(self.no_positions))[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(list(range(self.no_positions)), self.no_positions)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("Please specify positions order as a list")

        self.chkd_candidates = collections.OrderedDict()
        self.max_flag = max_flag
        #print("max_flag",max_flag)
        self.root = Node(value='R', children_values=self.atom_types, struct=[None]*self.no_positions)
        self.acc_threshold = 0.1                                                  
        self.get_reward = get_reward

        if expand_children == "all":
            self.expand_children = len(self.atom_types)
        elif isinstance(expand_children, int):
            if (expand_children > len(self.atom_types)) or (expand_children == 0):
                sys.exit("Please choose appropriate number of children to expand")
            else:
                self.expand_children = expand_children
        self.result = Result()      ##调用结果这个程序
        self.play_out = play_out
        if play_out_selection == "best":
            self.play_out_selection_mean = False
        elif play_out_selection =="mean":
            self.play_out_selection_mean = True
        else:
            sys.exit("Please set play_out_selection to either mean or best")



        if ucb == "best":
            self.ucb_mean = False
        elif ucb =="mean":
            self.ucb_mean = True
        else:
            sys.exit("Please set ucb to either mean or best")


    def _enumerate_cand(self, struct, size):
        structure = struct[:]
        chosen_candidates = []
        if self.atom_const is not None:
            for value_id in range(len(self.atom_types)):
                if structure.count(self.atom_types[value_id]) > self.atom_const[value_id]:
                    return chosen_candidates
            for pout in range(size):
                cand = structure[:]
                for value_id in range(len(self.atom_types)):
                    diff = self.atom_const[value_id] - cand.count(self.atom_types[value_id])
                    if diff != 0:
                        avl_pos = [i for i, x in enumerate(cand) if x is None]
                        to_fill_pos = np.random.choice(avl_pos, diff, replace=False)
                        for pos in to_fill_pos:
                            cand[pos] = self.atom_types[value_id]
                chosen_candidates.append(cand)
        else:
            for pout in range(size):
                cand = structure[:]
                avl_pos = [i for i, x in enumerate(cand) if x is None]
                for pos in avl_pos:
                    cand[pos] = np.random.choice(self.atom_types)
                chosen_candidates.append(cand)
        return chosen_candidates

    def one_hot_encode(self,space):
        no_atoms=len(self.atom_types)
        new_space = np.empty((space.shape[0], space.shape[1], no_atoms), dtype=int)
        for at_ind, at in enumerate(self.atom_types):
            one_hot = np.zeros(no_atoms, dtype=int)
            one_hot[at_ind] = 1
            new_space[space == at] = one_hot
        return new_space.reshape(space.shape[0],space.shape[1]*no_atoms)

    def _simulate(self, struct, lvl):
        if self.space is None:
            return self._enumerate_cand(struct,self.play_out)
  



    def _simulate_matrix(self, struct):
        structure = struct[:]
        chosen_candidates = []
        filled_pos = [i for i, x in enumerate(structure) if x is not None]
        filled_values = [x for i, x in enumerate(structure) if x is not None]
        sub_data = self.space[:, filled_pos]
        avl_candidates_idx = np.where(np.all(sub_data == filled_values, axis=1))[0]
        if len(avl_candidates_idx) != 0:
            if self.play_out <= len(avl_candidates_idx):
                chosen_idxs = np.random.choice(avl_candidates_idx, self.play_out)
            else:
                chosen_idxs = np.random.choice(avl_candidates_idx, len(avl_candidates_idx))
            for idx in chosen_idxs:
                chosen_candidates.append(list(self.space[idx]))
        return chosen_candidates




    def search(self, no_candidates=None, display=True):

                 
            
        printcount1=[]
        printcount2=[]
        printcount3=[]
        printcount4=[]
        printcount5=[]
        
        prev_len = 0
        prev_current = None
        round_no = 1
        if no_candidates is None :
            sys.exit("Please specify no_candidates")
        else:
            fidelity=0.5
            while  fidelity<0.7 or len(self.chkd_candidates) < no_candidates :
            #while len(self.chkd_candidates) < no_candidates and fidelity<sdfid :
                current = self.root.select(self.max_flag, self.ucb_mean)     ###select
                #print("current",current,current.level)
                
                if current.level == self.no_positions:
                    struct = current.struct[:]
                    if str(struct) not in self.chkd_candidates.keys():           #
                        e = self.get_reward(struct)                              #reward
                        self.chkd_candidates[str(struct)] = e
                    else:
                        e = self.chkd_candidates[str(struct)]
                    current.bck_prop(e)                                          ##bck_prop                                             
                else:
                    position = self.positions_order[current.level]
                    try_children = current.expand(position, self.expand_children)        #expand
                   
                    for try_child in try_children:
                        #print( try_child)
                        
                        
                        all_struct = self._simulate(try_child.struct,try_child.level)           #simulaten_playout=5
                        #if len(all_struct) != 0:
                           #print(all_struct)
                   
                        rewards = []
                        for struct in all_struct:
                           # print(struct)
                            printcount1.append(struct[0])
                            printcount2.append(struct[1])
                            printcount3.append(struct[2])
                            printcount4.append(struct[3])
                            printcount5.append(struct[4])
                            
                            
                            
                            if str(struct) not in self.chkd_candidates.keys():
                                e = self.get_reward(struct)
                                if e is not False:
                                    self.chkd_candidates[str(struct)] = e       #reward 
                            else:
                                e = self.chkd_candidates[str(struct)]
                            rewards.append(e)
                        rewards[:] = [x for x in rewards if x is not False]
                        
                        if len(rewards)!=0:
                            
                            if self.play_out_selection_mean:
                                best_e = np.mean(rewards)
                            else:
                                if self.max_flag:
                                    best_e = max(rewards)
                                else:
                                    #print("min")
                                    best_e = min(rewards)
                            try_child.bck_prop(best_e)                           ##bck_prop     
                        else:
                            print("len(rewards)=0")                                        
                            current.children[try_child.value] = None
                            all_struct = self._simulate(current.struct,current.level)      
                            rewards = []
                            for struct in all_struct:
                                if str(struct) not in self.chkd_candidates.keys():
                                    e = self.get_reward(struct)
                                    self.chkd_candidates[str(struct)] = e
                                else:
                                    e = self.chkd_candidates[str(struct)]
                                rewards.append(e)
                            if self.play_out_selection_mean:
                                best_e = np.mean(rewards)
                            else:
                                if self.max_flag:
                                    best_e = max(rewards)
                                else:
                                    best_e = min(rewards)
                            current.bck_prop(best_e)                             ##bck_prop
                            
                if (current == prev_current) and (len(self.chkd_candidates) == prev_len):
                    adjust_val = (no_candidates-len(self.chkd_candidates))/no_candidates
                    if adjust_val < self.acc_threshold:
                        adjust_val = self.acc_threshold
                    #adjust_c
                    current.adjust_c(adjust_val)  
                    
                prev_len = len(self.chkd_candidates)
                prev_current = current
                
                optimal_fx=max(iter(self.chkd_candidates.values()))    #################### 
                DS=[(ast.literal_eval(x), v) for (x,v) in self.chkd_candidates.items()]
                optimal_candidate = [k for (k, v) in DS if v == optimal_fx]
                fidelity=optimal_fx 
                
                De=40
                ss=np.array(optimal_candidate)
                obs=np.zeros(( 5), dtype=np.float64)
                for i in range(5):
                    #obs[i]=-0.2/(i+1)+struct[i]%De*0.01/(i+1)
                    #obs[i]=-0.4+struct[i]%De*0.02
                    obs[i]=-0.2+ss[0,i]%De*0.01
                    
              #  pathdesign=GroverSearchEfficient(6,28,5) 
                    
             #   energy,fidelity = pathdesign.evolution(obs) 
                 
             
                print(round_no,optimal_candidate,obs,optimal_fx)   ##########)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
                f = open(r'C:\Users\yuqinchen\Desktop\a'+str(self.data)+'a.txt','a+')
                f.writelines([str(fidelity)])
                f.writelines(['\n'])
                f.close()
                
                if round_no %10==0:
                   # print(Counter(printcount1))
                   # print(Counter(printcount2))
                    #print(Counter(printcount3))
                    #print(Counter(printcount4))
                    #print(Counter(printcount5))
                    
                    labels, values = zip(*Counter(printcount1).items())

                    indexes = np.arange(len(labels))
                    width = 1
                    
                    #plt.bar(indexes, values, width)
                    #plt.xticks(indexes + width * 0.5, labels)
                    #plt.show()
                    
                    #labels, values = zip(*Counter(printcount2).items())

                    #indexes = np.arange(len(labels))
                    #width = 1
                    
                    #plt.bar(indexes, values, width)
                    #plt.xticks(indexes + width * 0.5, labels)
                    #plt.show()
                    
                    #labels, values = zip(*Counter(printcount3).items())

                    #indexes = np.arange(len(labels))
                    #width = 1
                    
                    #plt.bar(indexes, values, width)
                    #plt.xticks(indexes + width * 0.5, labels)
                    #plt.show()
                    
                    #labels, values = zip(*Counter(printcount4).items())

                   # indexes = np.arange(len(labels))
                   # width = 1
                    
                   # plt.bar(indexes, values, width)
                   # plt.xticks(indexes + width * 0.5, labels)
                  #  plt.show()
                    
                  #  labels, values = zip(*Counter(printcount5).items())

                  #  indexes = np.arange(len(labels))
                  #  width = 1
                    
                  #  plt.bar(indexes, values, width)
                  #  plt.xticks(indexes + width * 0.5, labels)
                  #  plt.show()
                                    
                
                
                
                #if display:
                #    print ("round ", round_no)
                ##    print ("checked candidates = ", len(self.chkd_candidates))
                #    if self.max_flag:
                #        print ("current best = ", max(iter(self.chkd_candidates.values())))
                #    else:
                #        print ("current best = ", min(iter(self.chkd_candidates.values())))
                round_no += 1
        self.result.format(no_candidates=no_candidates, chkd_candidates=self.chkd_candidates, max_flag=self.max_flag)
        self.result.no_nodes, visits, self.result.max_depth_reached = self.root.get_info()
        self.result.avg_node_visit = visits / self.result.no_nodes
        return self.result
