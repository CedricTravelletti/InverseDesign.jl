import numpy as np
from ase.build import bulk
import torch
import itertools
import gpytorch
from matplotlib import pyplot as plt
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.model import FantasizeMixin
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor
from typing import NoReturn, Optional
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.module import Module
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.transforms.input import InputTransform
from gpytorch.means.mean import Mean
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
import copy

## Define the dGP to include forces
class GPWithDerivatives(GPyTorchModel, gpytorch.models.ExactGP, FantasizeMixin):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        
        # Dimension of model
        dim = train_X.shape[-1]
        self.train_X = train_X 
        # Multi-dimensional likelihood since we're modeling a function and its gradient
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1+dim)
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
        self._num_outputs = 1+dim
        # Used to extract function value and not gradients during optimization
        self.scale_tensor = torch.tensor([1.0] + [0.0]*dim, dtype=torch.double)
        
    def forward(self, x):
        mean_x = self.mean_module(x).to(device)
        covar_x = self.covar_module(x).to(device)
        print(mean_x.shape)
        print(covar_x.shape)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ## Set up evaluation function (pipe to ASE) for trial parameters suggested by Ax. 
# Note that this function can return additional keys that can be used in the `outcome_constraints` of the experiment.

from botorch.models import SingleTaskGP, ModelListGP, FixedNoiseGP,MultiTaskGP
from botorch.acquisition import qMaxValueEntropy,qKnowledgeGradient, qNoisyExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound,ExpectedImprovement
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.surrogate import Surrogate
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# ## Create client and initial sampling strategy to warm-up the GP model
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

# Define a class for the experiment global experiment with everything included inside
import time
import os
from ase.optimize import BFGS
from ase.io import read, write
import pandas as pd
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.exceptions.core import OptimizationShouldStop
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import matplotlib.pyplot as plt
from ase.calculators.emt import EMT
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.core.metric import Metric
from sklearn.neighbors import KNeighborsRegressor
from Plotting import plotter
from settup import experiment_setup

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class BOExperiment(plotter, experiment_setup):
    number_of_exp = 0
    ExperimentNames = []
    BFGS_en = []
    BFGS_runtime = []
    BO_en = []
    BO_runtime = []

    def __init__(self, expname):
        self.expname = expname
        self.input = self.input_()
        self.atoms = bulk(self.input['surface_atom'], self.input['lattice'])
        self.gs = self.generation_stragegy()
        self.file_name_format = f"ase_ads_DF_{self.input['number_of_ads']}{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_" + "".join([char for char in self.input['bo_surrogate'] if char.isupper()]) + "_" + "".join([char for char in self.input['bo_acquisition_f'] if char.isupper()]) + f"_{self.expname}_*.csv"
        self.curr_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name = f"ASE_ads_{self.input['number_of_ads']}{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_" + "".join([char for char in self.input['bo_surrogate'] if char.isupper()]) + "_" + "".join([char for char in self.input['bo_acquisition_f'] if char.isupper()]) + f"_{self.curr_date_time}_{self.expname}"
        self.best_parameters = None
        self.best_objective = None
        self.best_parameters_list = []
        self.best_objective_list = []
        self.best_parameters = None
        self.best_objective = None
        self.best_parameters_list = []
        self.best_objective_list = []
        self.build_save_path()
        BOExperiment.add_exp()
        BOExperiment.ExperimentNames.append(self.expname)

    def BFGS_opt(self):
        #self.atoms.calc = getattr(sys.modules[__name__], self.input['calc_method'])
        self.atoms.calc = EMT() # Fix to be able to change in the future
        self.opt = BFGS(self.atoms, logfile=f'{self.folder_name}/BFGS.log', trajectory=f'{self.folder_name}/BFGS.traj')
        self.BFGS_start = time.time()
        self.opt.run(fmax=0.05)
        self.BFGS_runtime = time.time() - self.BFGS_start
        
        #XYZ file for blender visualization
        write(f'{self.folder_name}/BFGS_opt.xyz', read(f'{self.folder_name}/BFGS.traj', index=':'))
        self.df_bfgs = pd.read_csv(f'{self.folder_name}/BFGS.log', skiprows=0, sep='\s+')
        
        # change df_bfgs['Time'] from str to runtime from first run in seconds
        self.df_bfgs['Time'] = pd.to_datetime(self.df_bfgs['Time'])
        self.df_bfgs['Time'] = self.df_bfgs['Time'].dt.strftime('%H:%M:%S')
        self.df_bfgs['Time'] = pd.to_timedelta(self.df_bfgs['Time'])
        self.df_bfgs['Time'] = self.df_bfgs['Time'].dt.total_seconds()
        self.df_bfgs['Time'] = self.df_bfgs['Time'] - self.df_bfgs['Time'][0]
        BOExperiment.BFGS_runtime.append(self.df_bfgs['Time'][-1])
        
        self.BFGSparams = []
        self.BFGSparams.append(self.atoms[-1].position.copy())
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                self.BFGSparams.append(self.atoms[-i].position.copy())
        self.bfgs_en = self.atoms.get_potential_energy()
        #print(self.BFGSparams)
        BOExperiment.BFGS_en.append(self.bfgs_en)

    def build_save_path(self):
        #Create the folder for the experiment
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.file_path_BFGSlog = f'{self.folder_name}/BFGS.log'
        ## Check if the BFGS log file exists already, delete if it does
        if os.path.exists(self.file_path_BFGSlog):
            # Delete the file
            os.remove(self.file_path_BFGSlog)
            print(f"File '{self.file_path_BFGSlog}' has been deleted.")
        else:
            print(f"File '{self.file_path_BFGSlog}' does not exist.")
        
        self.save_path = f"{self.folder_name}/{self.file_name_format}"
        return self.save_path

    def conv_study(self):
        # To add later
        pass

    def BO(self):
        #Setup experiment
        self.setexp()
        self.atoms.calc = EMT()
        #Run Optimization loop
        self.start = time.time()
        self.run_time = []
        self.N_BO_steps = self.input['n_bo_steps']
        
        self.BO_trace_space_log_x = []
        self.BO_trace_space_log_y = []
        self.BO_trace_space_log_z = []
        
        if self.input['number_of_ads'] != 1:
            self.BO_trace_space_log_x2 = []
            self.BO_trace_space_log_y2 = []
            self.BO_trace_space_log_z2 = []
        
        self.BO_models = []
        
        if self.input['opt_stop'] == 'T':
            #run this if we want to stop the optimization after a threshold
            for i in range(self.N_BO_steps):
                if i >= 6:
                    self.BO_models.append(copy.deepcopy(self.ax_client.generation_strategy.model))
                try: 
                    parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization
                except OptimizationShouldStop as exc:
                    print(exc.message)
                    break
                self.paratrialplot = parameters
                if i >= 5:
                    self.step_plotting(num_trial=i)
                # Local evaluation here can be replaced with deployment to external system.
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate_OOP(parameters))
                self.run_time.append(time.time() - self.start)
                if self.input['mult_p'] == 'T':
                    #Store current BO trajectory
                    self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
                    print(type(self.params))
                    print(self.params)
                    
                    self.BO_trace_space_log_x.append(self.params[0]['x'])
                    self.BO_trace_space_log_y.append(self.params[0]['y'])
                    self.BO_trace_space_log_z.append(self.params[0]['z'])
                    
                    if self.input['number_of_ads'] != 1:
                        self.BO_trace_space_log_x2.append(self.params[0]['x2'])
                        self.BO_trace_space_log_y2.append(self.params[0]['y2'])
                        self.BO_trace_space_log_z2.append(self.params[0]['z2'])
                elif self.input['mult_p'] == 'F':
                    self.params = self.ax_client.get_best_parameters()[:1][0]
                    print(type(self.params))
                    print(self.params)
                    
                    self.BO_trace_space_log_x.append(self.params['x'])
                    self.BO_trace_space_log_y.append(self.params['y'])
                    self.BO_trace_space_log_z.append(self.params['z'])
                    
                    if self.input['number_of_ads'] != 1:
                        self.BO_trace_space_log_x2.append(self.params['x2'])
                        self.BO_trace_space_log_y2.append(self.params['y2'])
                        self.BO_trace_space_log_z2.append(self.params['z2'])
            if self.input['mult_p'] == 'T':
                BOExperiment.BO_en.append(self.params[1][0]['adsorption_energy'])
            elif self.input['mult_p'] == 'F':
                BOExperiment.BO_en.append(self.ax_client.get_best_parameters()[1][0]['adsorption_energy'])
            
                
            
                
            
        elif self.input['opt_stop'] == 'F':
            #run this if we want to run the optimization for a fixed number of steps
            for i in range(self.N_BO_steps):
                if i >= 6:
                    self.BO_models.append(copy.deepcopy(self.ax_client.generation_strategy.model))
                parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization, still not sure how to make it work
                self.paratrialplot = parameters
                if i >= 5:
                    self.step_plotting(num_trial=i)
                # Local evaluation here can be replaced with deployment to external system.
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate_OOP(parameters))
                self.run_time.append(time.time() - self.start)
                #Store current BO trajectory
                if self.input['mult_p'] == 'T':
                    #Store current BO trajectory
                    self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
                    print(type(self.params))
                    print(self.params)
                    
                    self.BO_trace_space_log_x.append(self.params[0]['x'])
                    self.BO_trace_space_log_y.append(self.params[0]['y'])
                    self.BO_trace_space_log_z.append(self.params[0]['z'])
                    
                    if self.input['number_of_ads'] != 1:
                        self.BO_trace_space_log_x2.append(self.params[0]['x2'])
                        self.BO_trace_space_log_y2.append(self.params[0]['y2'])
                        self.BO_trace_space_log_z2.append(self.params[0]['z2'])
                    
                elif self.input['mult_p'] == 'F':
                    self.params = self.ax_client.get_best_parameters()[:1][0]
                    print(type(self.params))
                    print(self.params)
                    
                    self.BO_trace_space_log_x.append(self.params['x'])
                    self.BO_trace_space_log_y.append(self.params['y'])
                    self.BO_trace_space_log_z.append(self.params['z'])
                    
                    if self.input['number_of_ads'] != 1:
                        self.BO_trace_space_log_x2.append(self.params['x2'])
                        self.BO_trace_space_log_y2.append(self.params['y2'])
                        self.BO_trace_space_log_z2.append(self.params['z2'])
            if self.input['mult_p'] == 'T':
                BOExperiment.BO_en.append(self.params[1][0]['adsorption_energy'])
            elif self.input['mult_p'] == 'F':
                BOExperiment.BO_en.append(self.ax_client.get_best_parameters()[1][0]['adsorption_energy'])
            
                
            
        BOExperiment.BO_runtime.append(self.run_time[-1])

        self.plot_acqf()

    def setexp(self):
        self.bulk_z_max = np.max(self.atoms[:-self.n_ads].positions[:, 2]) #modified to account for changes in initial conditions + universal
        self.cell_x_min, self.cell_x_max = float(np.min(self.atoms.cell[:, 0])), float(np.max(self.atoms.cell[:, 0]))
        self.cell_y_min, self.cell_y_max = float(np.min(self.atoms.cell[:, 1])), float(np.max(self.atoms.cell[:, 1]))
        self.z_adsorb_max = np.max([self.atoms[-1].position[-1], self.atoms[-2].position[-1]]) # modified to account for changes in initial conditions
        
        if self.input['mult_p'] == 'T':
            self.objectives={"adsorption_energy": ObjectiveProperties(minimize=True), "dx": ObjectiveProperties(minimize=True), "dy": ObjectiveProperties(minimize=True), "dz": ObjectiveProperties(minimize=True)}
            #, "dx2": ObjectiveProperties(minimize=True), "dy2": ObjectiveProperties(minimize=True), "dz2": ObjectiveProperties(minimize=True)
        elif self.input['mult_p'] == 'F':
            self.objectives={"adsorption_energy": ObjectiveProperties(minimize=True)}
        if self.input['opt_stop'] == 'T':
            self.stopping_strategy = ImprovementGlobalStoppingStrategy(
            min_trials=5 + 5, window_size=5, improvement_bar=0.01
            )
            self.ax_client = AxClient(generation_strategy=self.gs, global_stopping_strategy=self.stopping_strategy)
        elif self.input['opt_stop'] == 'F':
            self.ax_client = AxClient(generation_strategy=self.gs)
            
        if self.input['number_of_ads'] == 2:
            self.parameters=[
            {
                "name": "x",
                "type": "range",
                "bounds": [float(self.cell_x_min), float(self.cell_x_max)/2],
                "value_type": "float",
            },
            {
                "name": "y",
                "type": "range",
                "bounds": [float(self.cell_y_min), float(self.cell_y_max)/2],
                "value_type": "float",
            },
            {
                "name": "z",
                "type": "range",
                "bounds": [float(self.bulk_z_max), float(self.bulk_z_max + self.input['adsorbant_init_h'])],
                "value_type": "float",
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [float(self.cell_x_min), float(self.cell_x_max)/2],
                "value_type": "float",
            },
            {
                "name": "y2",
                "type": "range",
                "bounds": [float(self.cell_y_min), float(self.cell_y_max)/2],
                "value_type": "float",
            },
            {
                "name": "z2",
                "type": "range",
                "bounds": [float(self.bulk_z_max), float(self.z_adsorb_max)],
                "value_type": "float",
            },
            ]
        elif self.input['number_of_ads'] == 1:
            self.parameters=[
            {
                "name": "x",
                "type": "range",
                "bounds": [float(self.cell_x_min), float(self.cell_x_max)/2],
                "value_type": "float",
            },
            {
                "name": "y",
                "type": "range",
                "bounds": [float(self.cell_y_min), float(self.cell_y_max)/2],
                "value_type": "float",
            },
            {
                "name": "z",
                "type": "range",
                "bounds": [float(self.bulk_z_max), float(self.z_adsorb_max)],
                "value_type": "float",
            },
            ]
        self.ax_client.create_experiment(
            name="adsorption_experiment",
            parameters=self.parameters,
            #parameter_constraints=["((x-x2)**2+(y-y2)**2+(z-z2)**2)**(1/2) <= 2"],  # For n_ads = 2
            #objectives={"adsorption_energy": ObjectiveProperties(minimize=True), "dx": ObjectiveProperties(minimize=True), "dy": ObjectiveProperties(minimize=True), "dz": ObjectiveProperties(minimize=True), "dx2": ObjectiveProperties(minimize=True), "dy2": ObjectiveProperties(minimize=True), "dz2": ObjectiveProperties(minimize=True)},
            objectives=self.objectives,
            # outcome_constraints=["l2norm <= 1.25"],  # Optional.
        )

    def load_exp(self, exp_name): #Load previous experiment
        #To do later
        pass

    def evaluate_OOP(self, parameters): # 2 adsorbates find a way to combine the two later ?
        x = torch.tensor([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")], device = device)
        xp = []
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                xp.append(torch.tensor([parameters.get(f"x{i}"), parameters.get(f"y{i}"), parameters.get(f"z{i}")], device = device))
        #x2 = torch.tensor([parameters.get(f"x2"), parameters.get(f"y2"), parameters.get(f"z2")], device = device)
        # Can put zeros since constraints are respected by set_positions.
        if self.input['number_of_ads'] == 1:
            new_pos = torch.vstack([torch.zeros((len(self.atoms) - self.input['number_of_ads'], 3), device = device), x])
        if self.input['number_of_ads'] != 1:
            new_pos = torch.vstack([torch.zeros((len(self.atoms) - self.input['number_of_ads'], 3), device = device), x])
            for i in range(2,self.input['number_of_ads']+1):
                new_pos = torch.vstack([new_pos, xp[i-2]])
        self.atoms.set_positions(new_pos.cpu().numpy(), apply_constraint=True)
        self.energy = torch.tensor(self.atoms.get_potential_energy(), device=device)
        if self.input['mol_soft_constraint'] == 'T':
            if self.atoms.get_distance(-1, -2) > 2.0:
                print(f'mol. distance : {self.atoms.get_distance(-1, -2)}')
                self.energy = self.energy + (self.atoms.get_distance(-1, -2)-2.0)**3
                print(f'energy : {self.energy}')
        if self.input['number_of_ads'] == 1:
            dx,dy,dz = torch.abs(torch.tensor(self.atoms.get_forces()[-1], device = device))
        if self.input['number_of_ads'] != 1:
            dx,dy,dz = torch.abs(torch.tensor(self.atoms.get_forces()[-2]))
            dx2,dy2,dz2 = torch.abs(torch.tensor(self.atoms.get_forces()[-1]))
        # In our case, standard error is 0, since we are computing a synthetic function.
        if self.input['mult_p'] == 'T':
            if self.input['number_of_ads'] != 1:
                return {"adsorption_energy": (self.energy, 0.0), "dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0), "dx2": (dx2, 0.0), "dy2": (dy2, 0.0), "dz2": (dz2, 0.0)} # We have 0 noise on the target.
            elif self.input['number_of_ads'] == 1:
                return {"adsorption_energy": (self.energy, 0.0), "dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0)} # We have 0 noise on the target.
        elif self.input['mult_p'] == 'F':
            return {"adsorption_energy": (self.energy, 0.0)} # We have 0 noise on the target.

    def evaluate_OOP_mol3(self, parameters): #To Do later
        x = torch.tensor([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")], device = device) # Position of C
        # Need a second angle, the one between the plane defined by the molecule and the plane defined by the surface --> How 
        # to define this ?? cross or dot product?
        # Make O1 follow C x1 = ...
        # Make O2 follow C x2 = ...
        # Set angle between O1-C-O2 = parameters.get(f"angle") something like that
        # Set new C position and O1 and O2 follow
        new_pos = torch.vstack([torch.zeros((len(self.atoms) - self.input['number_of_ads'], 3), device = device), x])
        # new angle = ;set_angle ...
        self.atoms.set_positions(new_pos.cpu().numpy(), apply_constraint=True)
        # new angle = ;set_angle ...
        self.energy = torch.tensor(self.atoms.get_potential_energy(), device=device)
        #dx,dy,dz = torch.abs(torch.tensor(self.atoms.get_forces()[-2], device = device))
        dx,dy,dz = torch.tensor(self.atoms.get_forces()[-2], device = device)
        #dTheta = 
        # In our case, standard error is 0, since we are computing a synthetic function.
        return {"adsorption_energy": (self.energy, 0.0), "dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0), "dTheta": (dTheta, 0.0)} # We have 0 noise on the target.
        #return {"adsorption_energy": (energy, 0.0),"dx": (dx, 0.0), "dy": (dy, 0.0), "dz": (dz, 0.0), "dx2": (dx2, 0.0), "dy2": (dy2, 0.0), "dz2": (dz2, 0.0)} # We have 0 noise on the target.

    def save_exp_json(self):#Save experiment and Ax client as JSON file
        self.ax_client.save_to_json_file(filepath= f"{self.folder_name}/ase_ads_DF_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.json")

    def save_exp_csv(self):    #Save experiment as csv file
        #Build result dataframe
        self.df = self.ax_client.get_trials_data_frame()
        # Get the trace only accounting for adsorption energy
        self.optimization_config = OptimizationConfig(objective=Objective(metric=Metric("adsorption_energy"), minimize=True)) # Base the trace on adsorption energy and not the default which is a combination of all objectives
        self.df['bo_trace'] = self.ax_client.get_trace(optimization_config=self.optimization_config)
        self.df["run_time"] = self.run_time
        self.df["BFGS_runtime"] = self.BFGS_runtime        
        
        self.df['opt_bfgs_x'],self.df['opt_bfgs_y'],self.df['opt_bfgs_z'] = self.BFGSparams[0]
        
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                self.df['opt_bfgs_x'+str(i)]= self.BFGSparams[i-1][0]
                self.df['opt_bfgs_y'+str(i)]= self.BFGSparams[i-1][1]
                self.df['opt_bfgs_z'+str(i)]= self.BFGSparams[i-1][2]
        #self.df['opt_bfgs_x2']= self.BFGS_params2[0]
        #self.df['opt_bfgs_y2']= self.BFGS_params2[1]
        #self.df['opt_bfgs_z2']= self.BFGS_params2[2]
        
        if self.input['mult_p'] == 'T':
            self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
            self.df['opt_bo_x']= self.params[0]['x']
            self.df['opt_bo_y']= self.params[0]['y']
            self.df['opt_bo_z']= self.params[0]['z']
            if self.input['number_of_ads'] != 1:
                self.df['opt_bo_x2']= self.params[0]['x2']
                self.df['opt_bo_y2']= self.params[0]['y2']
                self.df['opt_bo_z2']= self.params[0]['z2']
                self.df['opt_bo_energy'] = self.params[1][0]['adsorption_energy']
        elif self.input['mult_p'] == 'F':
            self.params = self.ax_client.get_best_parameters()[:1][0]
            self.df['opt_bo_x']= self.params['x']
            self.df['opt_bo_y']= self.params['y']
            self.df['opt_bo_z']= self.params['z']
            if self.input['number_of_ads'] != 1:
                self.df['opt_bo_x2']= self.params['x2']
                self.df['opt_bo_y2']= self.params['y2']
                self.df['opt_bo_z2']= self.params['z2']
            self.df['opt_bo_energy'] = self.ax_client.get_best_parameters()[1][0]['adsorption_energy']
        self.df['opt_bfgs_energy'] = self.bfgs_en        
        
        #Save results as dataframe csv file
        self.dfname = f"{self.folder_name}/ase_ads_DF_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.csv"
        
        #Save BO trajectory as dataframe csv file
        if self.input['number_of_ads'] == 1:
            self.df_bo_space_trace = pd.DataFrame(list(zip(self.BO_trace_space_log_x, self.BO_trace_space_log_y, self.BO_trace_space_log_z)), columns =['x', 'y', 'z'])
        elif self.input['number_of_ads'] != 1:
            self.df_bo_space_trace = pd.DataFrame(list(zip(self.BO_trace_space_log_x, self.BO_trace_space_log_y, self.BO_trace_space_log_z,self.BO_trace_space_log_x2, self.BO_trace_space_log_y2, self.BO_trace_space_log_z2)), columns =['x', 'y', 'z','x2', 'y2', 'z2'])
        
        self.df_bo_space_trace_name = f"{self.folder_name}/ase_ads_DF_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_BO_space_trace_{self.curr_date_time}.csv"
        
        if self.input["save_fig"] == "T":
            self.df.to_csv(self.dfname, index=False)
            self.df_bo_space_trace.to_csv(self.df_bo_space_trace_name, index=False)

    def generation_stragegy(self):
        gs = GenerationStrategy(
            steps=[
                # 1. Initialization step (does not require pre-existing data and is well-suited for
                # initial sampling of the search space)
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=self.input['gs_init_steps'],  # How many trials should be produced from this generation step
                    min_trials_observed=3,  # How many trials need to be completed to move to next model
                    max_parallelism=5,  # Max parallelism for this step
                    #model_kwargs={},  # Any kwargs you want passed into the model
                    #model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                ),
                # 2. Bayesian optimization step (requires data obtained from previous phase and learns
                # from all data available at the time of each new candidate generation call)
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # No limitation on how many trials should be produced from this step
                    max_parallelism=3,
                    model_kwargs={
                        "torch_device": device,
                        "surrogate": Surrogate(
                                                # BoTorch `Model` type
                                                botorch_model_class=str_to_class(self.input['bo_surrogate']),
                                                # Optional, MLL class with which to optimize model parameters
                                                mll_class=ExactMarginalLogLikelihood,
                                                # Optional, dictionary of keyword arguments to underlying
                                                # BoTorch `Model` constructor
                                                model_options={"torch_device": device},
                                                ),
                        #"botorch_acqf_class": str_to_class(self.input['bo_acquisition_f']),
                        "botorch_acqf_class": str_to_class(self.input['bo_acquisition_f']),
                        #"acquisition_options": {"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*self.input["number_of_ads"]*3, dtype=torch.double))},
                        #"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*input["number_of_ads"]*3, dtype=torch.double)),
                                },  # Any kwargs you want passed into the model
                    model_gen_kwargs={"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*self.input["number_of_ads"]*3, dtype=torch.double))},
                    # Parallelism limit for this step, often lower than for Sobol
                    # More on parallelism vs. required samples in BayesOpt:
                    # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
                ),
            ]
        )
        return gs

    @classmethod
    def mean_BFGS(cls):
        return np.mean(cls.BFGS_en), np.mean(cls.BFGS_runtime)

    @classmethod
    def mean_BO(cls):
        return np.mean(cls.BO_en), np.mean(cls.BO_runtime)

    @classmethod
    def response_surface_fit(cls):
        # load a txt file as a list on floats
        def load_txt(filename):
            with open(filename) as f:
                data = f.readlines()
            data = [float(x.strip()) for x in data]
            return data
        trainx = load_txt('x_response_surface.txt')
        trainy = load_txt('y_response_surface.txt')
        positions = list(itertools.product(trainx, trainy))
        ##load response_surface.txt
        energies = load_txt('response_surface.txt')
        y = np.array(energies)
        ##
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(positions, y)
        y_pred = knn.predict(positions)
        ##Plot the predicted energies
        E_pred = y_pred.reshape(len(trainx),len(trainy), order = 'F')
        plt.contourf(trainx, trainy, E_pred, levels=100, label = 'Energy')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    @classmethod
    def record_data(cls):
        #To do later
        pass

    @classmethod
    def add_exp(cls):
        cls.number_of_exp += 1
