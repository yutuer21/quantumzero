
"""
Created on Fri Oct 25 14:46:36 2019

@author: yuqinchen
"""
from dynamics import SatSearchEfficient
import mcts
import math
from numpy import *
from scipy import linalg
import numpy as np

import random
from qutip import * 

class system():
    
    def satSystem(n_qubit,result):
        N=2**n_qubit
    
    
        sx=np.array([[0,1],[1,0]])
        si=np.array([[1,0],[0,1]])
    
        HB=np.kron(si,si)
        for i in range(n_qubit-2):
            HB=np.kron(si,HB)
        HB=n_qubit*HB/2
    
        for j in range(n_qubit):
            if j==0:
                B=sx
                for i in range(n_qubit-1-j):
                    B=np.kron(si,B) 
            else:    
                for i in range(j):
                    if i==0:
                        B=si 
                    else: 
                        B=np.kron(si,B)  
                B=np.kron(sx,B)   
                for i in range(n_qubit-1-j):
                    B=np.kron(si,B)
    
            HB=HB-B/2
    
    
    
        
        HC= np.zeros((N,N))
        for ii in result:
            i=int(ii)
            HC[i,i]=HC[i,i]+1
        HP=HC
        

        
        bb=[np.sqrt(1./2**n_qubit)]*2**n_qubit
        sit=HP.tolist().index(min(HP.tolist()))
        bp=[0]*2**n_qubit
        bp[sit]=1
        
        psi0 =Qobj(np.array(bb))
        psif =Qobj(np.array(bp))
        
        
        return HB,HP,psi0,psif
    


class method():
    
    
    def linear(n_qubit,T,Mcut,HB,HP,psi0,psif):
        pathdesign=SatSearchEfficient(n_qubit,T,Mcut,HB,HP,psi0,psif)   
    
        obs=np.array([0.,0., 0.,0., 0.])
        energy,fidelity =  pathdesign.evolution(obs)
        return energy,fidelity
    
   
    def StochasticDescent(n_qubit,T,Mcut,HB,HP,psi0,psif):
        pathdesign=SatSearchEfficient(n_qubit,T,Mcut,HB,HP,psi0,psif)
        
        delta=0.01#/4
        iterNum=100
        ncan=0
        obs=np.random.rand(Mcut)*0.   #0.02
        list = [x/100  for x in range(-20, 20,1)]
       

        for i in range (Mcut):
        #  #  obs[i]=random.uniform(-0.2,0.2)
            slice = random.sample(list, 1) 
            obs[i]=slice[0]
        print(obs)
        
        iter = 0
        while True:
            iter += 1
            
            energy,fidelity = pathdesign.evolution(obs)
            ncan=ncan+1
            obs1=obs
            fid=fidelity
            num_converge = 0
            for m in range(1, 1+Mcut):
                #print('Updating parameter b{}'.format(m))
                if  obs[m-1]+ delta > 0.2 or obs[m-1] - delta < -0.2:
                    num_converge += 1
                    continue
                obs[m-1] += delta #/m    #高频的变化幅度大于低频
                #print(delta/m )
                #print(obs)
                energy,fidelity =  pathdesign.evolution(obs)
                ncan=ncan+1
                if fidelity > fid:
                    fid=fidelity
                    #print('b{}+delta'.format(m))
                    continue 
                else:
                    obs[m-1] -= 2*delta #/m   
                    #print(delta/m )
                    #print(obs)
                    energy,fidelity =  pathdesign.evolution(obs)       
                    ncan=ncan+1
                    if fidelity > fid:
                        fid=fidelity
                        #print('b{}-delta'.format(m))
                        continue 
                    else:
                        obs[m-1] += delta #/m
                        num_converge += 1  
                        #print('keep invariant')  
                        continue 
            if num_converge == Mcut:
                #print('fidelity:', fid,energy)
                break
            if iter > iterNum:
                print('WARNING: The algorithm does not converge')
                break 
            #print("ss:",obs)
        #print(iter)
        print("iter:",iter,"ncan:",ncan)
        return obs1, fid
    
    
    
    
    
    
    
    
    def mcts(data,n_qubit,T,Mcut,HB,HP,psi0,psif,ncandidates):
        
        pathdesign=SatSearchEfficient(n_qubit,T,Mcut,HB,HP,psi0,psif)
        
        def get_reward(struct):
            delta=0.1             ##update lenth
            De=40  #20
            Mcut=5 #5                   ## frequence cut
        
            obs=np.zeros((Mcut), dtype=np.float64)   
            for i in range(Mcut):
                obs[i]=-0.2+struct[i]%De*0.01
                
            energy,fidelity = pathdesign.evolution(obs)    
            cond=fidelity  
        
            return cond
        
        myTree=mcts.Tree(data,T,no_positions=5, atom_types=list(range(40)), atom_const=None, get_reward=get_reward, positions_order=list(range(5)),
                max_flag=True,expand_children=10, play_out=5, play_out_selection="best", space=None, candidate_pool_size=100,
                 ucb="mean")
        
        res=myTree.search(display=True,no_candidates=ncandidates)

        print (res.checked_candidates_size/50)
        fidelity=res.optimal_fx  
        obs=res.optimal_candidate 
        
        return obs,fidelity