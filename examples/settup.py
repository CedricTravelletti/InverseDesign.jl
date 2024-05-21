from ase.build import bulk
import torch

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Define a class for the experiment global experiment with everything included inside
from ase.eos import calculate_eos
from ase.build import add_adsorbate, fcc111
import random
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.eam import EAM
from ase.collections import g2

class experiment_setup:
    def review_input(self):
        return print(self.input)

    def input_(self):
        self.input = {}
        #Read input file
        with open('Input.txt', 'r') as file:
            for line in file:
                # Ignore lines starting with '#'
                if line.startswith('#'):
                    continue
                key, value = line.strip().split(' : ')
                try:
                    # Convert to integer if possible
                    self.input[key] = int(value)
                except ValueError:
                    # If not possible, store as string
                    self.input[key] = value
        if self.input['number_of_ads'] == 1:
            self.input['mol_soft_constraint'] = 'F'
            self.input['plotted_atom'] = 0
        if self.input['number_of_ads'] != 1:
            self.input['ads_init_pos'] = 'random'
        return self.input

    def set_cells(self):
        self.atoms = bulk(self.input['surface_atom'], self.input['lattice'])
        #self.atoms.calc = getattr(sys.modules[__name__], self.input['calc_method'])
        self.atoms.calc = EMT() # Fix to be able to change in the future
        self.atoms.EOS = calculate_eos(self.atoms)
        self.atoms.v, self.atoms.e, self.atoms.B = self.atoms.EOS.fit()
        self.atoms.cell *= (self.atoms.v / self.atoms.get_volume())**(1/3)
        return self.atoms.get_potential_energy()

    def pre_ads(self):
        self.a = self.atoms.cell[0, 1] * 2
        self.n_layers = self.input['number_of_layers']
        self.atoms = fcc111(self.input['surface_atom'], (self.input['supercell_x_rep'], self.input['supercell_y_rep'], self.n_layers), a=self.a)

    def add_adsorbant(self):
        self.pre_ads()
        self.ads_height = float(self.input['adsorbant_init_h'])
        self.n_ads = self.input['number_of_ads']
        self.ads = self.input['adsorbant_atom']
        for i in range(self.n_ads): #not yet supported by BO, supported for BFGS -- now supported by BO for 1-2 atoms
            if self.input['ads_init_pos'] == 'random':
                self.poss = (self.a + self.a*random.random()*self.input['supercell_x_rep']/2, self.a + self.a*random.random()*self.input['supercell_y_rep']/2)
            else:
                self.poss = self.input['ads_init_pos']
            add_adsorbate(self.atoms, self.ads, height=self.ads_height, position=self.poss)
        self.atoms.center(vacuum = self.input['supercell_vacuum'], axis = 2) 
        
        # Constrain all atoms except the adsorbate:
        self.fixed = list(range(len(self.atoms) - self.n_ads))
        self.atoms.constraints = [FixAtoms(indices=self.fixed)]

    def add_molecule(self):
        self.pre_ads()
        self.molecule = g2[self.input['adsorbant_molecule']]
        self.n_ads = len(self.molecule)
        self.ads_height = float(self.input['adsorbant_init_h']) #TO CHANGE
        if self.input['ads_init_pos'] == 'random':
            self.poss = (self.a + self.a*random.random()*self.input['supercell_x_rep']/2, self.a + self.a*random.random()*self.input['supercell_y_rep']/2)
        else:
            self.poss = self.input['ads_init_pos']
        add_adsorbate(self.atoms, self.molecule, height=self.ads_height, position=self.poss)
        self.atoms.center(vacuum = self.input['supercell_vacuum'], axis = 2) 
        # Constrain all atoms except the adsorbate:
        self.fixed = list(range(len(self.atoms) - len(self.molecule)))
        self.atoms.constraints = [FixAtoms(indices=self.fixed)]
