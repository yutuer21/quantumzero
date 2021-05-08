"""
Created on Fri Oct 25 14:46:36 2019

@author: yuqinchen
"""

import numpy as np 
from qutip import * 
from numpy import *



class SatSearchEfficient():
    def __init__(self, n_qubit, T, Mcut,HB,HP,psi0,psif):
        self.n_qubit = n_qubit
        self.N = 2**self.n_qubit
        self.T = T 
        self.Mcut = Mcut
        
        self.psi0 =psi0
        self.psif =psif

        self.H1=Qobj(HB)
        self.H2=Qobj(HP)
  
      

    def H1_coef(self, t, args):
        s = t/self.T 
        j = 1
        for b in args:
            s += args[b] * np.sin(j*np.pi*t/self.T)
            j += 1
        return 1-s

    def H2_coef(self, t, args):
        s = t/self.T 
        j = 1
        for b in args:
            s += args[b] * np.sin(j*np.pi*t/self.T)
            j += 1
        return s

    def evolution(self, bstate):
        """
        bstate: 1D array
        
        Return energy expectation and fidelity
        """
        args = {}
        for i, b in enumerate(list(bstate)):
            args['b{}'.format(i+1)] = b 
            
            
        dt=0.5
        NL=self.T/dt
    
         
        
        t = np.linspace(dt, self.T-dt, int(NL))
        
        H = [[self.H1, self.H1_coef], [self.H2, self.H2_coef]]
        output = mesolve(H, self.psi0, t, args=args)
      
        
        states = output.states[-1]
        
        c= (self.psif.dag()) * states    ####overlap
        fidelity=np.abs(c[0,0])**2
       
        x=states.dag()*self.H2*states      ###energy
        energy=x[0,0]
        
    
        return energy, fidelity


    def fidelity(self, args):
        H = [[self.H1, self.H1_coef], [self.H2, self.H2_coef]]
        t = np.linspace(0, self.T, 100)
        output = mesolve(H, self.psi0, t, e_ops=self.obs, args=args)
        fidelity = output.expect[0][-1]
        return fidelity


