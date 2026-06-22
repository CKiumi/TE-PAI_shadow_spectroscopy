
"""
Python Module to do TE_PAI shadow spectroscopy 
based on the following articles:  "Algorithmic Shadow Spectroscopy" from  "Hans Hon Sang Chan", "Richard Meister", "Matthew L. Goh", "Bálint Koczor"
"TE-PAI: Exact Time Evolution by Sampling Random Circuits" from "Chusei Kiumi" and "Bálint Koczor"

The module te_pai, Shadow_Spectroscopy, Hamiltonian , Hardware simulationcan all be used independantly from the others. 
One should be aware of How hamiltonian are defined in this project: 
We define any Hamiltonian that can be write as a linear combinaison  of pauli operation as : 
[("Pauli", [qubit], coef),...]

exemple : 
[("ZZ", [0,1], -2),("ZZ", [1,2], -2),("X", [0], -0.1)("X", [1], -0.1)("X", [2], -0.1)]

is equivalent to : 
H=-2(Z0Z1+Z1Z2)-0.1(X0+X1+X2)
where Pn is the P pauli applied to the n qubits

The main, and the tests need the use of the "Quantum_tools" folder


"""

__version__="1.0.0"

__date__="13/05/2025"

__author__="Hugo PAGES"

__email__="hugo.pages@etu.unistra.fr"

__license__="MIT License "  
       
    
__articles__ = [
    {
    "title": "TE-PAI: Exact Time Evolution by Sampling Random Circuits",
    "author": ["Chusei Kiumi", "Bálint Koczor"],
    "year": 2024,
    "eprint": "2410.16850",
    "archivePrefix": "arXiv",
    "primaryClass": "quant-ph",
    "url": "https://arxiv.org/abs/2410.16850"
},
    {
        "title": "Algorithmic Shadow Spectroscopy",
        "author": ["Hans Hon Sang Chan", "Richard Meister", "Matthew L. Goh", "Bálint Koczor"],
        "year": 2024,
        "eprint": "2212.11036",
        "archivePrefix": "arXiv",
        "primaryClass": "quant-ph",
        "url": "https://arxiv.org/abs/2212.11036"
    }
]

from .hamiltonian import *
from .te_pai import TE_PAI
from .shadow_spectro import ClassicalShadow, ShadowSpectro, Spectroscopy
from .te_pai_shadow.te_pai_shadow_spectroscopy import TE_PAI_Shadow_Spectroscopy 