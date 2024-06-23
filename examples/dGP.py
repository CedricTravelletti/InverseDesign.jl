# Imports
import os
import itertools

import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient, qNoisyExpectedImprovement, UpperConfidenceBound, PosteriorMean, ExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model import FantasizeMixin
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.gp_regression import SingleTaskGP

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from typing import Any, Union
from gpytorch import settings as gpt_settings
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.containers import BotorchContainer

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Set default data type to avoid numerical issues from low precision
torch.set_default_dtype(torch.double)

# Set seed for consistency and good results
seed = 541
torch.manual_seed(seed)
np.random.seed(seed)

"""
Setup for the Bayesian Optimization
"""

from ase.build import bulk
from ase.eos import calculate_eos
from ase.build import add_adsorbate, fcc111
import random
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.eam import EAM
from ase.collections import g2

# Define the input parameters
input = {}
#Read input file
with open('examples/Input.txt', 'r') as file:
    for line in file:
        # Ignore lines starting with '#'
        if line.startswith('#'):
            continue
        key, value = line.strip().split(' : ')
        try:
            # Convert to integer if possible
            input[key] = int(value)
        except ValueError:
            # If not possible, store as string
            input[key] = value
if input['number_of_ads'] == 1:
    input['mol_soft_constraint'] = 'F'
    input['plotted_atom'] = 0
if input['number_of_ads'] != 1:
    input['ads_init_pos'] = 'random'
input['mult_p'] = 'T'

atoms = bulk(input['surface_atom'], input['lattice'])
#atoms.calc = getattr(sys.modules[__name__], input['calc_method'])
atoms.calc = EMT() # Fix to be able to change in the future
atoms.EOS = calculate_eos(atoms)
atoms.v, atoms.e, atoms.B = atoms.EOS.fit()
atoms.cell *= (atoms.v / atoms.get_volume())**(1/3)

a = atoms.cell[0, 1] * 2
n_layers = input['number_of_layers']
atoms = fcc111(input['surface_atom'], (input['supercell_x_rep'], input['supercell_y_rep'], n_layers), a=a)

ads_height = float(input['adsorbant_init_h'])
n_ads = input['number_of_ads']
ads = input['adsorbant_atom']
for i in range(n_ads): #not yet supported by BO, supported for BFGS -- now supported by BO for 1-2 atoms
    if input['ads_init_pos'] == 'random':
        poss = (a + a*random.random()*input['supercell_x_rep']/2, a + a*random.random()*input['supercell_y_rep']/2)
    else:
        poss = input['ads_init_pos']
    add_adsorbate(atoms, ads, height=ads_height, position=poss)
atoms.center(vacuum = input['supercell_vacuum'], axis = 2) 

# Constrain all atoms except the adsorbate:
fixed = list(range(len(atoms) - n_ads))
atoms.constraints = [FixAtoms(indices=fixed)]
atoms.calc = EMT()

"""
Define the GpyTorch model
"""
# Define the dGP
class GPWithDerivatives(GPyTorchModel, gpytorch.models.ExactGP, FantasizeMixin):
    def __init__(self, train_X, train_Y):
        # Dimension of model
        dim = train_X.shape[-1] 
        # Multi-dimensional likelihood since we're modeling a function and its gradient
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1 + dim)
        super(GPWithDerivatives, self).__init__(train_X, train_Y, likelihood)
        # Gradient-enabled mean
        self.mean_module = gpytorch.means.ConstantMeanGrad() 
        # Gradient-enabled kernel
        self.base_kernel = gpytorch.kernels.RBFKernelGrad( 
            ard_num_dims=dim, # Separate lengthscale for each input dimension
        )
        # Adds lengthscale to the kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        # Output dimension is 1 (function value) + dim (number of partial derivatives)
        self._num_outputs = 1 + dim
        # Used to extract function value and not gradients during optimization
        self.scale_tensor = torch.tensor([1.0] + [0.0]*dim, dtype=torch.double)
        train_X = train_X.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
#    def posterior(self, X, observation_noise = False, **kwargs: Any) -> GPyTorchPosterior:
#        with gpt_settings.fast_pred_var(False):
#            posterior = super().posterior(X=X, observation_noise=observation_noise, **kwargs)
#        return posterior

"""
Define the evaluation function
"""

def evaluate_OOP(parameters): # 2 adsorbates find a way to combine the two later ?
        #Get the trial positions from the model; and set the atoms object to these positions
        x = torch.tensor([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")], device = device)
        xp = []
        if input['number_of_ads'] != 1:
            for i in range(2,input['number_of_ads']+1):
                xp.append(torch.tensor([parameters.get(f"x{i}"), parameters.get(f"y{i}"), parameters.get(f"z{i}")], device = device))
        
        new_pos = torch.vstack([torch.zeros((len(atoms) - input['number_of_ads'], 3), device = device), x])
        
        if input['number_of_ads'] != 1:
            for i in range(2,input['number_of_ads']+1):
                new_pos = torch.vstack([new_pos, xp[i-2]])
        
        atoms.set_positions(new_pos.cpu().numpy(), apply_constraint=True)
        
        #Compute the energy of the system with the current trial positions
        energy = torch.tensor(atoms.get_potential_energy(), device=device)
        
        # Add here Soft constraint for a 2 atom molecule
        # --------------------------------------------- #
        # --------------------------------------------- #
        #if input['mol_soft_constraint'] == 'T':
        #    if atoms.get_distance(-2, -1) > 2.0:
        #        print(f'mol. distance : {atoms.get_distance(-2, -1)}')
        #        energy = energy + (atoms.get_distance(-2, -1)-2.0)**input['soft_constraint_power']
        #        print(f'Molecule atom distance: {atoms.get_distance(-2, -1)} Angstrom')
        #        print(f'Soft constraint energy term: {(atoms.get_distance(-2, -1)-2.0)**input["soft_constraint_power"]}')
        # --------------------------------------------- #
        # --------------------------------------------- #
        # --------------------------------------------- #
        
        
        # Add here Soft constraint for a 3 atom molecule
        # --------------------------------------------- #
        # --------------------------------------------- #
        if input['mol_soft_constraint'] == 'T':
            if atoms.get_distance(-3, -2) > 2.0:
                print(f'mol. distance : {atoms.get_distance(-3, -2)}')
                energy = energy + (atoms.get_distance(-3, -2)-2.0)**input['soft_constraint_power']
                print(f'Molecule atom distance: {atoms.get_distance(-3, -2)} Angstrom')
                print(f'Soft constraint energy term: {(atoms.get_distance(-3, -2)-2.0)**input["soft_constraint_power"]}')
            if atoms.get_distance(-3, -1) > 2.0:
                print(f'mol. distance : {atoms.get_distance(-3, -1)}')
                energy = energy + (atoms.get_distance(-3, -1)-2.0)**input['soft_constraint_power']
                print(f'Molecule atom distance: {atoms.get_distance(-3, -1)} Angstrom')
                print(f'Soft constraint energy term: {(atoms.get_distance(-3, -1)-2.0)**input["soft_constraint_power"]}')
        # --------------------------------------------- #
        # --------------------------------------------- #
        # --------------------------------------------- #
        
        #Compute the forces of the system with the current trial positions
        if input['mult_p'] == 'T':
            dx,dy,dz = [],[],[]
            dx.append(torch.tensor(atoms.get_forces()[-1][0], device = device))
            dy.append(torch.tensor(atoms.get_forces()[-1][1], device = device))
            dz.append(torch.tensor(atoms.get_forces()[-1][2], device = device))
            #dx = torch.tensor(atoms.get_forces()[-1][0], device = device)
            #dy = torch.tensor(atoms.get_forces()[-1][1], device = device)
            #dz = torch.tensor(atoms.get_forces()[-1][2], device = device)
            if input['number_of_ads'] != 1:
                for i in range(2,input['number_of_ads']+1):
                    dx.append(torch.tensor(atoms.get_forces()[-i][0], device = device))
                    dy.append(torch.tensor(atoms.get_forces()[-i][1], device = device))
                    dz.append(torch.tensor(atoms.get_forces()[-i][2], device = device))

        #Return the objective values for the trial positions
        # In our case, standard error is 0, since we are computing a synthetic function.
        #if input['mult_p'] == 'T':
        #    objective_return = {"adsorption_energy": (energy, 0.0), "dx": (dx[0], 0.0), "dy": (dy[0], 0.0), "dz": (dz[0], 0.0)}
        #    if input['number_of_ads'] != 1:
        #        for i in range(2,input['number_of_ads']+1):
        #            objective_return[f"dx{i}"] = (dx[i-1], 0.0)
        #            objective_return[f"dy{i}"] = (dy[i-1], 0.0)
        #            objective_return[f"dz{i}"] = (dz[i-1], 0.0)
        #    return objective_return
        #    
        #elif input['mult_p'] == 'F':
        #    return {"adsorption_energy": (energy, 0.0)} # We have 0 noise on the target.
        return energy,dx, dy, dz


"""
Initialize the SOBOL runs and compute the initial training evaluations
"""
#Train info corresponds to the SOBOL initialization
# Random search locations
bulk_z_max = np.max(atoms[:-n_ads].positions[:, 2]) #modified to account for changes in initial conditions + universal
cell_x_min, cell_x_max = float(np.min(atoms.cell[:, 0])), float(np.max(atoms.cell[:, 0]))
cell_y_min, cell_y_max = float(np.min(atoms.cell[:, 1])), float(np.max(atoms.cell[:, 1]))
z_adsorb_max = bulk_z_max + input['adsorbant_init_h'] # modified to account for changes in initial conditions

#seed = 580
#torch.manual_seed(seed)
#np.random.seed(seed)

sobol = SobolEngine(dimension=3, scramble=True)
samples_raw = sobol.draw(input['gs_init_steps']).to(device)
samples_raw

train_x = samples_raw[:,0]*(cell_x_max/2 - cell_x_min/2)+cell_x_min/2
train_y = samples_raw[:,1]*(cell_y_max/2 - cell_y_min/2)+cell_y_min/2
train_z = samples_raw[:,2]*(z_adsorb_max - bulk_z_max)+bulk_z_max


train_Y = torch.empty((input['gs_init_steps'], 4))
#Evaluate the function and its gradient at the random search locations
# Populate random search evaluations
for i in range(input['gs_init_steps']):
    obj,dx,dy,dz = evaluate_OOP(parameters={"x": train_x[i], "y": train_y[i], "z": train_z[i]})
    #obj, deriv = Rosenbrock(train_X[i])
    train_Y[i][0] = obj
    train_Y[i][1] = dx[0]
    train_Y[i][2] = dy[0]
    train_Y[i][3] = dz[0]

train_x = train_x.to(device)
train_y = train_y.to(device)
train_z = train_z.to(device)
train_Y = train_Y.to(device)
train_Y


"""
Run the BO loop
"""
mc_samples = 32 # Samples from Monte-Carlo sampler
import pandas as pd

for i in range(40):
    # Standardize domain and range, this prevents numerical issues
    mean_Y = train_Y.mean(dim=0)
    std_Y = train_Y.std(dim=0)
    unscaled_train_Y = train_Y
    scaled_train_Y = (train_Y - mean_Y) / std_Y
    
    mean_x = train_x.mean(dim=0)
    std_x = train_x.std(dim=0)
    unscaled_train_x = train_x
    scaled_train_x = (train_x - mean_x) / std_x
    stand_x = train_x / (cell_x_max/2)
    
    mean_y = train_y.mean(dim=0)
    std_y = train_y.std(dim=0)
    unscaled_train_y = train_y
    scaled_train_y = (train_y - mean_y) / std_y
    stand_y = train_y / (cell_y_max/2)
    
    mean_z = train_z.mean(dim=0)
    std_z = train_z.std(dim=0)
    unscaled_train_z = train_z
    scaled_train_z = (train_z - mean_z) / std_z
    stand_z = (train_z - bulk_z_max) / (z_adsorb_max-bulk_z_max)

    train = torch.vstack([train_x, train_y, train_z]).T
    scaled_train = torch.vstack([scaled_train_x, scaled_train_y, scaled_train_z]).T
    st_train = torch.vstack([stand_x, stand_y, stand_z]).T
    
    st_train.to(device)
    scaled_train.to(device)
    train.to(device)
    scaled_train_Y.to(device)
    # Initialize the dGP and fit it to the training data
    #dGP_model = GPWithDerivatives(scaled_train, scaled_train_Y) # Define dGP model
    dGP_model = GPWithDerivatives(scaled_train, scaled_train_Y).cuda() # Define dGP model
    mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model).cuda() # Define MLL
    fit_gpytorch_mll(mll, max_attempts=100).cuda()

    # Extract only the function value from the multi-output GP, the dGP
    scal_transf = ScalarizedPosteriorTransform(weights=dGP_model.scale_tensor).cuda()
    #scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*1, dtype=torch.double))

    # Create qNEI acquisition function
    sampler = SobolQMCNormalSampler(mc_samples).cuda()
    qNEI = qNoisyExpectedImprovement(dGP_model,
                scaled_train,
                sampler,
                posterior_transform=scal_transf).cuda()
    #qKG = qKnowledgeGradient(dGP_model,
    #            posterior_transform=scal_transf,
    #            num_fantasies=16)
    #UCB = UpperConfidenceBound(dGP_model, posterior_transform=scal_transf, beta=0.1)
    #Set bounds for optimization
    scaled_cell_x_min, scaled_cell_x_max = (cell_x_min/2 - mean_x)/std_x, (cell_x_max/2 - mean_x)/std_x
    scaled_cell_y_min, scaled_cell_y_max = (cell_y_min/2 - mean_y)/std_y, (cell_y_max/2 - mean_y)/std_y
    scaled_bulk_z_max, scaled_z_adsorb_max = (bulk_z_max - mean_z)/std_z, (z_adsorb_max - mean_z)/std_z

    bounds_ns = torch.vstack([torch.tensor([cell_x_min/2, cell_y_min/2, bulk_z_max]),
                            torch.tensor([cell_x_max/2, cell_y_max/2, z_adsorb_max])])
    bounds_ns = bounds_ns.to(device)
    bounds_st = torch.vstack([torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0,1.0,1.0])])
    
    # Rescale bounds based on training data
    bounds = torch.vstack([torch.tensor([scaled_cell_x_min, scaled_cell_y_min, scaled_bulk_z_max]),
                            torch.tensor([scaled_cell_x_max, scaled_cell_y_max, scaled_z_adsorb_max])])
    bounds = bounds.to(device)
    
    # Get candidate point for objective
    candidates, _ = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds.to(device),
        q=1,
        num_restarts=100,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 1, "maxiter": 1000},
    )

    # Rescale candidate back to original domain
    candidate = []
    candidate.append((candidates[0][0]  * std_x) + mean_x)
    candidate.append((candidates[0][1]  * std_y) + mean_y)
    candidate.append((candidates[0][2]  * std_z) + mean_z)
    
    #candidate.append((candidates[0][0] * cell_x_max/2))
    #candidate.append((candidates[0][1] * cell_y_max/2))
    #candidate.append((candidates[0][2] * (z_adsorb_max - bulk_z_max)) + bulk_z_max)

    # Evaluate the objective and add it to the list of data for the model
    obj, dx,dy,dz = evaluate_OOP(parameters={"x": candidate[0], "y": candidate[1], "z": candidate[2]})
    new_Y = torch.cat([obj.unsqueeze(0),dx[0].unsqueeze(0),dy[0].unsqueeze(0),dz[0].unsqueeze(0)])

    #Append evaluation to training data
    #add candidate[0] to train_x, candidate[1] to train_y, candidate[2] to train_z
    train_x = torch.hstack((train_x, candidate[0])).detach().clone()
    train_y = torch.hstack((train_y, candidate[1])).detach().clone()
    train_z = torch.hstack((train_z, candidate[2])).detach().clone()
    train_Y = torch.vstack((train_Y, new_Y)).detach().clone()
    print(f"Run: {i} done")


mean_Y = train_Y.mean(dim=0)
std_Y = train_Y.std(dim=0)
unscaled_train_Y = train_Y
scaled_train_Y = (train_Y - mean_Y) / std_Y

mean_x = train_x.mean(dim=0)
std_x = train_x.std(dim=0)
unscaled_train_x = train_x
scaled_train_x = (train_x - mean_x) / std_x
stand_x = train_x / (cell_x_max/2)

mean_y = train_y.mean(dim=0)
std_y = train_y.std(dim=0)
unscaled_train_y = train_y
scaled_train_y = (train_y - mean_y) / std_y
stand_y = train_y / (cell_y_max/2)

mean_z = train_z.mean(dim=0)
std_z = train_z.std(dim=0)
unscaled_train_z = train_z
scaled_train_z = (train_z - mean_z) / std_z
stand_z = (train_z - bulk_z_max) / (z_adsorb_max-bulk_z_max)
train = torch.vstack([train_x, train_y, train_z]).T
train.to(device)
scaled_train = torch.vstack([scaled_train_x, scaled_train_y, scaled_train_z]).T
scaled_train.to(device)
st_train = torch.vstack([stand_x, stand_y, stand_z]).T
st_train.to(device)
scaled_train_Y = scaled_train_Y.to(device)
# Initialize the dGP and fit it to the training data
#dGP_model = GPWithDerivatives(scaled_train, scaled_train_Y) # Define dGP model
dGP_model = GPWithDerivatives(scaled_train, scaled_train_Y).cuda() # Define dGP model
mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model).cuda() # Define MLL
fit_gpytorch_mll(mll, max_attempts=100).cuda()

#Store train a csv

train
train_np = train.cpu().detach().numpy()
train_df = pd.DataFrame(train_np)
train_df.to_csv('train_KG.csv', index=False)

train_Y
train_Y_np = train_Y.cpu().detach().numpy()
train_Y_df = pd.DataFrame(train_Y_np)
train_Y_df.to_csv('train_Y_KG.csv', index=False)


#train_Y
#min = np.argmin((train_Y[:,0]).detach().cpu().numpy())
#min
#train_Y[:,0]
#train[min]
#
#x_slice = train[min][0]
#y_slice = train[min][1]
#z_slice = train[min][2]

#After SOBOL
#qKG acqf plot for singel atom comparison

#GUI for file input
import tkinter as tk
from tkinter import filedialog
import pandas as pd
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

#df_EI = pd.read_csv(file_path)
train_sobol_STGP = df_EI[['x','y','z']].values
train_Y_sobol_STGP = df_EI[['adsorption_energy']].values
#add a column of zeros for the gradient
train_Y_sobol_STGP = np.hstack((train_Y_sobol_STGP, np.zeros((len(train_Y_sobol_STGP),1))))

#df_UCB = pd.read_csv(file_path)
train_sobol_STGP = df_UCB[['x','y','z']].values
train_Y_sobol_STGP = df_UCB[['adsorption_energy']].values
#add a column of zeros for the gradient
train_Y_sobol_STGP = np.hstack((train_Y_sobol_STGP, np.zeros((len(train_Y_sobol_STGP),1))))

df_KG = pd.read_csv(file_path)
train_sobol_STGP = df_KG[['x','y','z']].values
train_Y_sobol_STGP = df_KG[['adsorption_energy']].values
#add a column of zeros for the gradient
train_Y_sobol_STGP = np.hstack((train_Y_sobol_STGP, np.zeros((len(train_Y_sobol_STGP),1))))



train_sobol_STGP = [[3.191922129018775,2.263424733552034,14.293172228095031],[1.2633354365082172,1.3522717514968805,14.072578520459032], [1.4210030133728613,3.7601767513871365,14.803943130132652],
                    [4.301465882768051,0.40409047723715724,13.58529857074926],[4.630212585939519,4.285960091052166,13.827422571313477],[1.658317188656744,2.8137187571179845,14.67983763992504]]
train_Y_sobol_STGP = [[15.617666244506836,0],
[16.568634033203125,0],
[11.071893692016602,0],
[26.16341209411621,0],
[43.56285095214844,0],
[11.37975025177002,0]]

#UCB 6th run
#11.37975025177002,1.658317188656744,2.8137187571179845,14.67983763992504

#KG 6th run
#11.120584487915039,1.6371237306332296,2.940062769033164,14.814257137400888

#EI 6 run
#11.452106475830078,1.690740620159413,2.489335223437234,14.747838524785891

mean_Y_STGP = torch.tensor(train_Y_sobol_STGP).mean(dim=0)
std_Y_STGP = torch.tensor(train_Y_sobol_STGP).std(dim=0)
unscaled_train_Y_STGP = torch.tensor(train_Y_sobol_STGP)
scaled_train_Y_STGP = (torch.tensor(train_Y_sobol_STGP) - mean_Y_STGP) / std_Y_STGP
st_train_Y_STGP = torch.tensor(train_Y_sobol_STGP)
#of nan set to 0
scaled_train_Y_STGP[torch.isnan(scaled_train_Y_STGP)] = 0

mean_train_STGP = torch.tensor(train_sobol_STGP).mean(dim=0)
std_train_STGP = torch.tensor(train_sobol_STGP).std(dim=0)
unscaled_train_STGP = torch.tensor(train_sobol_STGP)
scaled_train_STGP = (torch.tensor(train_sobol_STGP) - mean_train_STGP) / std_train_STGP
st_train_STGP = torch.tensor(train_sobol_STGP)
st_train_STGP[:,0] = st_train_STGP[:,0] / (cell_x_max/2)
st_train_STGP[:,1] = st_train_STGP[:,1] / (cell_y_max/2)
st_train_STGP[:,2] = (st_train_STGP[:,2] - bulk_z_max) / (z_adsorb_max - bulk_z_max) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_train_STGP = st_train_STGP.to(device)
st_train_Y_STGP = scaled_train_Y_STGP.to(device)
scaled_train_Y_STGP = scaled_train_Y_STGP.to(device)

sampler = SobolQMCNormalSampler(32)

ST_GP_model = SingleTaskGP(st_train_STGP, scaled_train_Y_STGP)# Define dGP model
mll = ExactMarginalLogLikelihood(gpytorch.likelihoods.GaussianLikelihood(num_tasks=1), ST_GP_model).cuda() # Define MLL
fit_gpytorch_mll(mll, max_attempts=100).cuda()
scal_trans_STGP = ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*1, dtype=torch.double))
qKG_STGP = qKnowledgeGradient(ST_GP_model, num_fantasies=4, posterior_transform=scal_trans_STGP)
bestf =torch.min(scaled_train_Y_STGP[:,0])
EI_STGP = ExpectedImprovement(ST_GP_model, posterior_transform =scal_trans_STGP, maximize=False, best_f = bestf)
UCB = UpperConfidenceBound(ST_GP_model, posterior_transform = scal_trans_STGP, beta = 0.1, maximize=False)
bounds_st = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
#train_Y_dGP = torch.tensor([])
#for i in range(len(train_sobol_STGP)):
#    obj,dx,dy,dz = evaluate_OOP(parameters={"x": train_sobol_STGP[i][0], "y": train_sobol_STGP[i][1], "z": train_sobol_STGP[i][2]})
#    newY = torch.cat([obj.unsqueeze(0),dx[0].unsqueeze(0),dy[0].unsqueeze(0),dz[0].unsqueeze(0)])
#    train_Y_dGP = torch.cat([train_Y_dGP, newY.unsqueeze(0)])
#train_Y_dGP = torch.tensor(train_Y_dGP)
#
#mean_Y_dGP = train_Y_dGP.mean(dim=0)
#std_Y_dGP = train_Y_dGP.std(dim=0)
#unscaled_train_Y_dGP = train_Y_dGP
#scaled_train_Y_dGP = (train_Y_dGP - mean_Y_dGP) / std_Y_dGP
#st_train_Y_dGP = train_Y_dGP
#
#dGP_model = GPWithDerivatives(st_train_STGP, scaled_train_Y_dGP) # Define dGP model
#mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model) # Define MLL
#fit_gpytorch_mll(mll, max_attempts=100)
#scal_trans = ScalarizedPosteriorTransform(weights=dGP_model.scale_tensor)
#qKG_STGP = qKnowledgeGradient(dGP_model, num_fantasies=16, posterior_transform=scal_trans)
#bestf =torch.min(scaled_train_Y_dGP[:,0])
#EI_STGP = ExpectedImprovement(dGP_model, posterior_transform=scal_trans, maximize=False, best_f = bestf)
#UCB = UpperConfidenceBound(dGP_model, posterior_transform=scal_trans, beta=0.1, maximize=False)

#ACQF xyfix plot
zlin = np.linspace(bulk_z_max, z_adsorb_max, 50)
xlin = np.linspace(cell_x_min, cell_x_max/2, 50)
ylin = np.linspace(cell_y_min, cell_y_max/2, 50)

EI_pred = optimize_acqf(EI_STGP, q = 1, bounds = bounds_st, num_restarts = 100, raw_samples = 1024, options = {"batch_limit": 1, "maxiter": 2000})
UCB_pred = optimize_acqf(UCB, q = 1, bounds = bounds_st, num_restarts = 100, raw_samples = 1024, options = {"batch_limit": 1, "maxiter": 2000})

x_slice_UCB = UCB_pred[0][0][0] * (cell_x_max/2)
y_slice_UCB = UCB_pred[0][0][1] * (cell_y_max/2)
z_slice_UCB = UCB_pred[0][0][2] * (z_adsorb_max - bulk_z_max) + bulk_z_max

x_slice_EI = EI_pred[0][0][0] * (cell_x_max/2)
y_slice_EI = EI_pred[0][0][1] * (cell_y_max/2)
z_slice_EI = EI_pred[0][0][2] * (z_adsorb_max - bulk_z_max) + bulk_z_max

x_slice_qKG = 1.832273224102932 
y_slice_qKG = 3.595543913468192 
z_slice_qKG = 15.032464791491012 

#Use x_slice, y_slice, zlin
x_st_UCB = x_slice_UCB / (cell_x_max/2)
y_st_UCB = y_slice_UCB / (cell_y_max/2)
z_st_UCB = (zlin - bulk_z_max) / (z_adsorb_max - bulk_z_max)

x_st_EI = x_slice_EI / (cell_x_max/2)
y_st_EI = y_slice_EI / (cell_y_max/2)
z_st_EI = (zlin - bulk_z_max) / (z_adsorb_max - bulk_z_max)

x_st_qKG = x_slice_qKG / (cell_x_max/2)
y_st_qKG = y_slice_qKG / (cell_y_max/2)
z_st_qKG = (zlin - bulk_z_max) / (z_adsorb_max - bulk_z_max)

# Acqf plot for xy fixed
posz_qKG = torch.tensor(list(itertools.product([x_st_qKG],[y_st_qKG], z_st_qKG)), device=device)
posz_EI = torch.tensor(list(itertools.product([x_st_EI],[y_st_EI], z_st_EI)), device=device)
posz_UCB = torch.tensor(list(itertools.product([x_st_UCB],[y_st_UCB], z_st_UCB)), device=device)

qNEI_plot_data_z = EI_STGP(posz_EI.view(-1, 1, 3)).cuda()
UCB_plot_data_z = UCB(posz_UCB.view(-1, 1, 3)).cuda()
posz_qKG = posz_qKG.repeat(17,1,1).transpose(0,1)
qKG_plot_data_z = qKG_STGP.evaluate(posz_qKG, bounds = bounds_st).cuda()

plt.plot(zlin, qKG_plot_data_z.cpu().detach().numpy(), "b*", label = 'qKG-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'z0 [A]')
plt.legend()
plt.xlim(bulk_z_max, z_adsorb_max)
plt.title(f'BO Acquisition criterion vs z, x = {x_slice_qKG:.2f}, y = {y_slice_qKG:.2f}, trial 41, atom 0')
plt.show()

plt.plot(zlin, qNEI_plot_data_z.cpu().detach().numpy(), "r*", label = 'EI-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'z0 [A]')
plt.legend()
plt.xlim(bulk_z_max, z_adsorb_max)
plt.title(f'BO Acquisition criterion vs z, x = {x_slice_EI:.2f} y = {y_slice_EI:.2f}, trial 41, atom 0')
plt.show()

plt.plot(zlin, UCB_plot_data_z.cpu().detach().numpy(), "g*", label = 'UCB-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'z0 [A]')
plt.legend()
plt.xlim(bulk_z_max, z_adsorb_max)
plt.title(f'BO Acquisition criterion vs z, x = {x_slice_UCB:.2f}, y = {y_slice_UCB:.2f}, trial 41, atom 0')
plt.show()

#Acqf plot for xz fixed
z_st_EI = (z_slice_EI - bulk_z_max) / (z_adsorb_max - bulk_z_max)
z_st_UCB = (z_slice_UCB - bulk_z_max) / (z_adsorb_max - bulk_z_max)
z_st_qKG = (z_slice_qKG - bulk_z_max) / (z_adsorb_max - bulk_z_max)

y_st_EI = ylin / (cell_y_max/2)
y_st_UCB = ylin / (cell_y_max/2)
y_st_qKG = ylin / (cell_y_max/2)

posy_UCB = torch.tensor(list(itertools.product([x_st_UCB],y_st_UCB, [z_st_UCB])))
posy_EI = torch.tensor(list(itertools.product([x_st_EI],y_st_EI, [z_st_EI])))
posy_qKG = torch.tensor(list(itertools.product([x_st_qKG],y_st_qKG, [z_st_qKG]))).cuda()

qNEI_plot_data_y = EI_STGP(posy_EI.view(-1, 1, 3))
UCB_plot_data_y = UCB(posy_UCB.view(-1, 1, 3))
posy_qKG = posy_qKG.repeat(17,1,1).transpose(0,1)
qKG_plot_data_y = qKG_STGP.evaluate(posy_qKG, bounds = bounds_st).cuda()

plt.plot(ylin, qKG_plot_data_y.cpu().detach().numpy(),"b*", label = 'qKG-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'y0 [A]')
plt.legend()
plt.xlim(min(ylin), max(ylin))
plt.title(f'BO Acquisition criterion vs y, x = {x_slice_qKG:.2f} z = {z_slice_qKG:.2f}, trial 41, atom 0')
plt.show()

plt.plot(ylin, qNEI_plot_data_y.cpu().detach().numpy(), "r*", label = 'EI-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'y0 [A]')
plt.legend()
plt.xlim(min(ylin), max(ylin))
plt.title(f'BO Acquisition criterion vs y, x = {x_slice_EI:.2f}, z = {z_slice_EI:.2f}, trial 5, atom 0')
plt.show()

plt.plot(ylin, UCB_plot_data_y.cpu().detach().numpy(), "g*", label = 'UCB-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'y0 [A]')
plt.legend()
plt.xlim(min(ylin), max(ylin))
plt.title(f'BO Acquisition criterion vs y, x = {x_slice_UCB:.2f}, z = {z_slice_UCB:.2f}, trial 6, atom 0')
plt.show()

#Acqf plot for yz fixed
y_st_EI = (y_slice_EI ) / (cell_y_max/2)
y_st_UCB = (y_slice_UCB ) / (cell_y_max/2)
y_st_qKG = (y_slice_qKG ) / (cell_y_max/2)

x_st_EI = xlin / (cell_x_max/2)
x_st_UCB = xlin / (cell_x_max/2)
x_st_qKG = xlin / (cell_x_max/2)

posx_UCB = torch.tensor(list(itertools.product(x_st_UCB,[y_st_UCB], [z_st_UCB])))
posx_EI = torch.tensor(list(itertools.product(x_st_EI,[y_st_EI], [z_st_EI])))
posx_qKG = torch.tensor(list(itertools.product(x_st_qKG,[y_st_qKG], [z_st_qKG]))).cuda()
qNEI_plot_data_x = EI_STGP(posx_EI.view(-1, 1, 3))
UCB_plot_data_x = UCB(posx_UCB.view(-1, 1, 3))
posx_qKG = posx_qKG.repeat(17,1,1).transpose(0,1)
qKG_plot_data_x = qKG_STGP.evaluate(posx_qKG, bounds = bounds_st).cuda()

plt.plot(xlin, qKG_plot_data_x.cpu().detach().numpy(),"b*", label = 'qKG-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'x0 [A]')
plt.legend()
plt.xlim(min(xlin), max(xlin))
plt.title(f'BO Acquisition criterion vs x, y = {y_slice_qKG:.2f} z = {z_slice_qKG:.2f}, trial 41, atom 0')
plt.show()

plt.plot(xlin, qNEI_plot_data_x.cpu().detach().numpy(), "r*", label = 'EI-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'x0 [A]')
plt.legend()
plt.xlim(min(xlin), max(xlin))
plt.title(f'BO Acquisition criterion vs x, y = {y_slice_EI:.2f}, z = {z_slice_EI:.2f}, trial 41, atom 0')
plt.show()

plt.plot(xlin, UCB_plot_data_x.cpu().detach().numpy(), "g*", label = 'UCB-BO Model')
plt.grid()
plt.ylabel('Criterion Value')
plt.xlabel(f'x0 [A]')
plt.legend()
plt.xlim(min(xlin), max(xlin))
plt.title(f'BO Acquisition criterion vs x, y = {y_slice_UCB:.2f}, z = {z_slice_UCB:.2f}, trial 41, atom 0')
plt.show()


#Plot a contourf with z fixed
zlin = np.linspace(bulk_z_max, z_adsorb_max, 10)
xlin = np.linspace(cell_x_min, cell_x_max/2, 10)
ylin = np.linspace(cell_y_min, cell_y_max/2, 10)

x_st_EI = xlin / (cell_x_max/2)
y_st_EI = ylin / (cell_y_max/2)

x_st_UCB = xlin / (cell_x_max/2)
y_st_UCB = ylin / (cell_y_max/2)

x_st_qKG = xlin / (cell_x_max/2)
y_st_qKG = ylin / (cell_y_max/2)

z_st_EI = (z_slice_EI - bulk_z_max) / (z_adsorb_max - bulk_z_max)
z_st_UCB = (z_slice_UCB - bulk_z_max) / (z_adsorb_max - bulk_z_max)
z_st_qKG = (z_slice_qKG - bulk_z_max) / (z_adsorb_max - bulk_z_max)


X, Y = np.meshgrid(xlin, ylin)
pos_EI = torch.tensor(list(itertools.product(x_st_EI, y_st_EI, [z_st_EI])))
pos_UCB = torch.tensor(list(itertools.product(x_st_UCB, y_st_UCB, [z_st_UCB])))
pos_qKG = torch.tensor(list(itertools.product(x_st_qKG, y_st_qKG, [z_st_qKG]))).cuda()
qNEI_plot_data = EI_STGP(pos_EI.view(-1, 1, 3))
UCB_plot_data = UCB(pos_UCB.view(-1, 1, 3))
pos_qKG = pos_qKG.repeat(5,1,1).transpose(0,1)
qKG_plot_data = qKG_STGP.evaluate(pos_qKG, bounds = bounds_st).cuda()

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, qKG_plot_data.cpu().detach().numpy().reshape(10,10), 100, cmap='inferno_r')
fig.colorbar(CS, ax=ax)
plt.title(f'BO Acquisition criterion vs x,y, z = {z_slice_qKG:.2f}, trial 41, atom 0')
plt.xlabel('x0 [A]')
plt.ylabel('y0 [A]')
plt.show()

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, qNEI_plot_data.cpu().detach().numpy().reshape(50,50), 100, cmap='inferno')
fig.colorbar(CS, ax=ax)
plt.title(f'BO Acquisition criterion vs x,y, z = {(z_slice_EI):.2f}, trial 41, atom 0')
plt.xlabel('x0 [A]')
plt.ylabel('y0 [A]')
plt.show()

fig, ax = plt.subplots()
CS = ax.contourf(X, Y, UCB_plot_data.cpu().detach().numpy().reshape(50,50), 100, cmap='inferno')
fig.colorbar(CS, ax=ax)
plt.title(f'BO Acquisition criterion vs x,y, z = {z_slice_UCB:.2f}, trial 41, atom 0')
plt.xlabel('x0 [A]')
plt.ylabel('y0 [A]')
plt.show()




















import tkinter as tk
from tkinter import filedialog
import pandas as pd

root = tk.Tk()
root.withdraw()

file_path_UCB = filedialog.askopenfilename()
file_path_UCB_Y = filedialog.askopenfilename()
data_UCB = pd.read_csv(file_path_UCB, header=None)
train_xx = torch.tensor(data_UCB[0].values, device = device)
train_yy = torch.tensor(data_UCB[1].values, device = device)
train_zz = torch.tensor(data_UCB[2].values, device = device)
trainn = torch.vstack([train_xx, train_yy, train_zz]).T
train_Y = pd.read_csv(file_path_UCB_Y, header=None)
#Make every row of train_Y a tensor
train_Y = torch.tensor(train_Y.values, device = device)

mean_Y = train_Y.mean(dim=0)
std_Y = train_Y.std(dim=0)
scaled_train_Y = (train_Y - mean_Y) / std_Y

mean_x = trainn[:,0].mean(dim=0)
std_x = trainn[:,0].std(dim=0)
mean_y = trainn[:,1].mean(dim=0)
std_y = trainn[:,1].std(dim=0)
mean_z = trainn[:,2].mean(dim=0)
std_z = trainn[:,2].std(dim=0)

st_train = torch.vstack([trainn[:,0]/(cell_x_max/2), trainn[:,1]/(cell_y_max/2), (trainn[:,2] - bulk_z_max) / (z_adsorb_max - bulk_z_max)]).T
st_train.to(device)
scaled_trainn = torch.vstack([(trainn[:,0] - mean_x) *std_x, (trainn[:,1] - mean_y)*std_y, (trainn[:,2] - mean_z)*std_z]).T
scaled_trainn.to(device)
scaled_train_Y.to(device)

dGP_model = GPWithDerivatives(scaled_trainn, scaled_train_Y) # Define dGP model
mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model) # Define MLL
fit_gpytorch_mll(mll.to(device), max_attempts=100)



#Find the minimum
acqf = PosteriorMean(dGP_model, posterior_transform=ScalarizedPosteriorTransform(torch.tensor([1.0] + [0.0]*3, device = device)), maximize=False)

#class PosteriorMeanOfGradientGP(PosteriorMean):
#    def forward(self, X: torch.Tensor) -> torch.Tensor:
#        gp = self.model
#        X_var = X.clone().requires_grad_(True)
#        with torch.enable_grad():
#            y_pred = gp.likelihood(gp(X_var))
#            y_mean_sum = y_pred.mean.sum()
#            dy_dx = torch.autograd.grad(y_mean_sum, X_var, create_graph = True)[0]
#        return dy_dx.mean((-2, -1))
#acqf = PosteriorMeanOfGradientGP(model=dGP_model, posterior_transform=ScalarizedPosteriorTransform(torch.tensor([1.0] + [0.0]*3, device = device)), maximize=False)

a = optimize_acqf(acqf, bounds = bounds.to(device), q = 1, num_restarts = 100, raw_samples = 1024, options = {"batch_limit": 1, "maxiter": 2000})
dGP_model(a[0]).mean * std_Y + mean_Y

dGP_model.eval()

xx = np.linspace(cell_x_min/2, cell_x_max/2, 60)
yy = np.linspace(cell_y_min/2, cell_y_max/2, 60)
zz = np.linspace(bulk_z_max, z_adsorb_max, 60)

zzz = a[0][0][2].to(device) * std_z + mean_z
yyy = a[0][0][1].to(device) * std_y + mean_y
xxx = a[0][0][0].to(device) * std_x + mean_x

Eev = []
for i in range(len(xx)):
    obj, dx,dy,dz = evaluate_OOP(parameters={"x": xx[i], "y": yyy, "z": zzz})
    Eev.append(obj.item())

zzz = torch.ones(len(xx), device=device)*(zzz - mean_z)/std_z
yyy = torch.ones(len(xx), device=device)*(yyy- mean_y)/std_y
tensor = torch.empty(1,3, device = device)
for i in range(len(xx)):
    tensor = torch.vstack([tensor, torch.tensor([xx[i], yyy[i], zzz[i]], device = device)])

tensor[:,0] = (tensor[:,0] - mean_x) / std_x
tensor[:,1] = (tensor[:,1] - mean_y) / std_y
tensor[:,2] = (tensor[:,2] - mean_z) / std_z

posterior = dGP_model(tensor)
posterior.mean * std_Y + mean_Y

plt.plot(xx, Eev)
plt.plot(xx, (posterior.mean[1:,0] * std_Y[0] + mean_Y[0]).cpu().detach().numpy(), "b*")
plt.show()










#Contour plot of the posterior
xx = np.linspace(cell_x_min/2, cell_x_max/2, 30)
yy = np.linspace(cell_y_min/2, cell_y_max/2, 30)
zz = np.linspace(bulk_z_max, z_adsorb_max, 30)
#zzz = a[0][0][2].to(device) * std_z + mean_z
zzz = 14.6
pos = list(itertools.product(xx, yy, [zzz]))
pos = torch.tensor(pos, device = device)
pos[:,0] = (pos[:,0] - mean_x) / std_x
pos[:,1] = (pos[:,1] - mean_y) / std_y
pos[:,2] = (pos[:,2] - mean_z) / std_z

dGP_model.eval()
posterior = dGP_model(pos)

plt.contourf(xx, yy, (posterior.mean[:,0] * std_Y[0] + mean_Y[0]).cpu().detach().numpy().reshape(30,30),100)
plt.colorbar(label = 'Energy Surface')
plt.scatter(train[:,0].cpu().detach().numpy(), train[:,1].cpu().detach().numpy(), color = "black", label = "Trials", marker="x", alpha = 0.7)
plt.xlabel('x0 [A]')
plt.ylabel('y0 [A]')
plt.title(f"BO predicted x0-y0 Energy surface, trial 0, z = {zzz:.2f}")
plt.legend()
plt.show()

#Contour plot of the posterior
xx = np.linspace(cell_x_min/2, cell_x_max/2, 30)
yy = np.linspace(cell_y_min/2, cell_y_max/2, 30)
zz = np.linspace(bulk_z_max, z_adsorb_max, 30)
#xxx = a[0][0][0].to(device) * std_x + mean_x
xxx = 3
pos = list(itertools.product([xxx], yy, zz))
pos = torch.tensor(pos, device = device)
pos[:,0] = (pos[:,0] - mean_x) / std_x
pos[:,1] = (pos[:,1] - mean_y) / std_y
pos[:,2] = (pos[:,2] - mean_z) / std_z

dGP_model.eval()
posterior = dGP_model(pos)

plt.contourf(yy, zz, (posterior.mean[:,0] * std_Y[0] + mean_Y[0]).cpu().detach().numpy().reshape(30,30),100)
plt.colorbar(label = 'Energy Surface')
plt.scatter(train[:,1].cpu().detach().numpy(), train[:,2].cpu().detach().numpy(), color = "black", label = "Trials", marker="x", alpha = 0.7)
plt.xlabel('y0 [A]')
plt.ylabel('z0 [A]')
plt.title(f"BO predicted x0-y0 Energy surface, trial 0, x = {xxx:.2f}")
plt.legend()
plt.show()

#Contour plot of the posterior
xx = np.linspace(cell_x_min/2, cell_x_max/2, 30)
yy = np.linspace(cell_y_min/2, cell_y_max/2, 30)
zz = np.linspace(bulk_z_max, z_adsorb_max, 30)
#yyy = a[0][0][1].to(device) * std_y + mean_y
yyy = 1
pos = list(itertools.product(xx, [yyy], zz))
pos = torch.tensor(pos, device = device)
pos[:,0] = (pos[:,0] - mean_x) / std_x
pos[:,1] = (pos[:,1] - mean_y) / std_y
pos[:,2] = (pos[:,2] - mean_z) / std_z

dGP_model.eval()
posterior = dGP_model(pos)

plt.contourf(xx, zz, (posterior.mean[:,0] * std_Y[0] + mean_Y[0]).cpu().detach().numpy().reshape(30,30),100)
plt.colorbar(label = 'Energy Surface')
plt.scatter(train[:,0].cpu().detach().numpy(), train[:,2].cpu().detach().numpy(), color = "black", label = "Trials", marker="x", alpha = 0.7)
plt.xlabel('x0 [A]')
plt.ylabel('z0 [A]')
plt.title(f"BO predicted x0-y0 Energy surface, trial 0, y = {yyy:.2f}")
plt.legend()
plt.show()