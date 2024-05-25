# Imports
import os

import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt

from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient, qNoisyExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model import FantasizeMixin
from botorch.models.gpytorch import GPyTorchModel

from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood


# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Set default data type to avoid numerical issues from low precision
torch.set_default_dtype(torch.double)

# Set seed for consistency and good results
seed = 3
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



input = {}
#Read input file
with open('Input.txt', 'r') as file:
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
        super().__init__(train_X, train_Y, likelihood)
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

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


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


train_x = torch.rand((input['gs_init_steps']))*(cell_x_max/2 - cell_x_min/2)+cell_x_min/2
train_y = torch.rand((input['gs_init_steps']))*(cell_y_max/2 - cell_y_min/2)+cell_y_min/2
train_z = torch.rand((input['gs_init_steps']))*(z_adsorb_max - bulk_z_max)+bulk_z_max

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

train_Y


"""
Run the BO loop
"""
mc_samples = 32 # Samples from Monte-Carlo sampler
"""
Non scaled loop
"""
for i in range(30):
    train = torch.vstack([train_x, train_y, train_z]).T

    # Initialize the dGP and fit it to the training data
    dGP_model = GPWithDerivatives(train, train_Y) # Define dGP model
    mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model) # Define MLL
    fit_gpytorch_mll(mll, max_attempts=20)

    # Extract only the function value from the multi-output GP, the dGP
    scal_transf = ScalarizedPosteriorTransform(weights=dGP_model.scale_tensor)

    # Create qNEI acquisition function
    sampler = SobolQMCNormalSampler(mc_samples)
    qNEI = qNoisyExpectedImprovement(dGP_model,
                train,
                sampler,
                posterior_transform=scal_transf)
    #qKG = qKnowledgeGradient(dGP_model,
    #            posterior_transform=scal_transf,
    #            num_fantasies=5)
    
    #Set bounds for optimization
    bounds_ns = torch.vstack([torch.tensor([cell_x_min/2, cell_y_min/2, bulk_z_max]),
                            torch.tensor([cell_x_max/2, cell_y_max/2, z_adsorb_max])])
    # Get candidate point for objective
    candidates, _ = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds_ns,
        q=1,
        num_restarts=100,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 1, "maxiter": 1000},
    )

    # Rescale candidate back to original domain
    candidate = []
    #candidate.append((candidates[0][1]  * std_x) + mean_x)
    #candidate.append((candidates[0][1]  * std_y) + mean_y)
    #candidate.append((candidates[0][2]  * std_z) + mean_z)
    
    candidate.append(candidates[0][1])
    candidate.append(candidates[0][1])
    candidate.append(candidates[0][2])

    # Evaluate the objective and add it to the list of data for the model
    obj, dx,dy,dz = evaluate_OOP(parameters={"x": candidate[0], "y": candidate[1], "z": candidate[2]})
    new_Y = torch.cat([obj.unsqueeze(0),dx[0].unsqueeze(0),dy[0].unsqueeze(0),dz[0].unsqueeze(0)])

    #Append evaluation to training data
    #add candidate[0] to train_x, candidate[1] to train_y, candidate[2] to train_z
    train_x = torch.hstack((train_x, candidate[0])).detach().clone()
    train_y = torch.hstack((train_y, candidate[1])).detach().clone()
    train_z = torch.hstack((train_z, candidate[2])).detach().clone()
    train_Y = torch.vstack((train_Y, new_Y)).detach().clone()


#Plotting
xx = np.linspace(cell_x_min/2, cell_x_max/2, 100)
yy = np.linspace(cell_y_min/2, cell_y_max/2, 100)
zz = np.linspace(bulk_z_max, z_adsorb_max, 100)

zzz = 14.6
yyy = 2.5

Eev = []
for i in range(len(xx)):
    obj, dx,dy,dz = evaluate_OOP(parameters={"x": xx[i], "y": yyy, "z": zzz})
    Eev.append(obj.item())

# Make a tensor with different values for x, but fixed y and z
zzz = torch.ones(len(xx))*zzz
yyy = torch.ones(len(xx))*yyy
tensor = torch.empty(1,3)
for i in range(len(xx)):
    tensor = torch.vstack([tensor, torch.tensor([xx[i], yyy[i], zzz[i]])])

dGP_model.eval()
posterior = dGP_model(tensor)
posterior.mean

plt.plot(tensor[:,0].cpu().numpy(), posterior.mean[:,0].cpu().detach().numpy(), "b*")
plt.plot(xx, Eev)
plt.show()

posterior.mean[:,0]












for i in range(15):
    # Standardize domain and range, this prevents numerical issues
    mean_Y = train_Y.mean(dim=0)
    std_Y = train_Y.std(dim=0)
    unscaled_train_Y = train_Y
    scaled_train_Y = (train_Y - mean_Y) / std_Y
    
    mean_x = train_x.mean(dim=0)
    std_x = train_x.std(dim=0)
    unscaled_train_x = train_x
    scaled_train_x = (train_x - mean_x) / std_x
    
    mean_y = train_y.mean(dim=0)
    std_y = train_y.std(dim=0)
    unscaled_train_y = train_y
    scaled_train_y = (train_y - mean_y) / std_y
    
    mean_z = train_z.mean(dim=0)
    std_z = train_z.std(dim=0)
    unscaled_train_z = train_z
    scaled_train_z = (train_z - mean_z) / std_z

    train = torch.vstack([train_x, train_y, train_z]).T
    scaled_train = torch.vstack([scaled_train_x, scaled_train_y, scaled_train_z]).T

    # Initialize the dGP and fit it to the training data
    dGP_model = GPWithDerivatives(scaled_train, scaled_train_Y) # Define dGP model
    mll = ExactMarginalLogLikelihood(dGP_model.likelihood, dGP_model) # Define MLL
    fit_gpytorch_mll(mll, max_attempts=20)

    # Extract only the function value from the multi-output GP, the dGP
    scal_transf = ScalarizedPosteriorTransform(weights=dGP_model.scale_tensor)

    # Create qNEI acquisition function
    sampler = SobolQMCNormalSampler(mc_samples)
    qNEI = qNoisyExpectedImprovement(dGP_model,
                train,
                sampler,
                posterior_transform=scal_transf)
    #qKG = qKnowledgeGradient(dGP_model,
    #            posterior_transform=scal_transf,
    #            num_fantasies=5)
    
    #Set bounds for optimization
    scaled_cell_x_min, scaled_cell_x_max = (cell_x_min/2 - mean_x)/std_x, (cell_x_max/2 - mean_x)/std_x
    scaled_cell_y_min, scaled_cell_y_max = (cell_y_min/2 - mean_y)/std_y, (cell_y_max/2 - mean_y)/std_y
    scaled_bulk_z_max, scaled_z_adsorb_max = (bulk_z_max - mean_z)/std_z, (z_adsorb_max - mean_z)/std_z

    bounds_ns = torch.vstack([torch.tensor([cell_x_min/2, cell_y_min/2, bulk_z_max]),
                            torch.tensor([cell_x_max/2, cell_y_max/2, z_adsorb_max])])

    # Rescale bounds based on training data
    bounds = torch.vstack([torch.tensor([scaled_cell_x_min, scaled_cell_y_min, scaled_bulk_z_max]),
                            torch.tensor([scaled_cell_x_max, scaled_cell_y_max, scaled_z_adsorb_max])])

    # Get candidate point for objective
    candidates, _ = optimize_acqf(
        acq_function=qNEI,
        bounds=bounds,
        q=1,
        num_restarts=100,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 1, "maxiter": 1000},
    )

    # Rescale candidate back to original domain
    candidate = []
    candidate.append((candidates[0][1]  * std_x) + mean_x)
    candidate.append((candidates[0][1]  * std_y) + mean_y)
    candidate.append((candidates[0][2]  * std_z) + mean_z)

    # Evaluate the objective and add it to the list of data for the model
    obj, dx,dy,dz = evaluate_OOP(parameters={"x": candidate[0], "y": candidate[1], "z": candidate[2]})
    new_Y = torch.cat([obj.unsqueeze(0),dx[0].unsqueeze(0),dy[0].unsqueeze(0),dz[0].unsqueeze(0)])

    #Append evaluation to training data
    #add candidate[0] to train_x, candidate[1] to train_y, candidate[2] to train_z
    train_x = torch.hstack((train_x, candidate[0])).detach().clone()
    train_y = torch.hstack((train_y, candidate[1])).detach().clone()
    train_z = torch.hstack((train_z, candidate[2])).detach().clone()
    train_Y = torch.vstack((train_Y, new_Y)).detach().clone()
