import numpy as np
from ase.build import bulk
import torch
import itertools
import gpytorch
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

from typing import List
from typing import Any, Dict

from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import settings as gpt_settings
from typing import Any, Union
from gpytorch.models import ExactGP
from botorch.utils.containers import BotorchContainer

## Define the dGP to include forces
class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @classmethod
    def construct_inputs(cls, training_data: BotorchContainer, **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}


class GPWithDerivatives(GPyTorchModel, ExactGP,FantasizeMixin):
    
    _num_outputs = 4  # to inform GPyTorchModel API
    
    def __init__(self, train_X: Tensor, train_Y: Tensor,):
        # Dimension of model
        dim = train_X.shape[-1]
        self.dim = dim
        self.train_X = train_X
        self.train_Y = train_Y
        # Multi-dimensional likelihood since we're modeling a function and its gradient
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1 + dim)
        super(GPWithDerivatives, self).__init__(train_X, train_Y.squeeze(-1), likelihood)
        
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=dim) # Separate lengthscale for each input dimension
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
        self.scale_tensor = torch.tensor([1.0] + [0.0] * dim, dtype=torch.double)

    def forward(self, x):
        print(f"x: {x}")
        print(self.train_X, self.train_Y)
        mean_x = self.mean_module(x).to(device)
        covar_x = self.covar_module(x).to(device)
        print(f"mean_x: {mean_x}")
        print(f"covar_x: {covar_x}")     
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: BotorchContainer, **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}


# ## Set up evaluation function (pipe to ASE) for trial parameters suggested by Ax. 
# Note that this function can return additional keys that can be used in the `outcome_constraints` of the experiment.

from botorch.models import SingleTaskGP, ModelListGP, FixedNoiseGP,MultiTaskGP,HeteroskedasticSingleTaskGP
from botorch.acquisition import qMaxValueEntropy,qKnowledgeGradient, qNoisyExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound,ExpectedImprovement
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.surrogate import Surrogate

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
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



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
        
        #Strore the BFGS optimized positions for all the adsorbates
        self.BFGSparams = []
        self.BFGSparams.append(self.atoms[-1].position.copy())
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                self.BFGSparams.append(self.atoms[-i].position.copy())
                
        #Get the optimized BFGS energy, store it with the other experiments
        self.bfgs_en = self.atoms.get_potential_energy()
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
        self.BO_models = []
        
        for i in range(self.N_BO_steps):
            if i >= 6:
                self.BO_models.append(copy.deepcopy(self.ax_client.generation_strategy.model))
            if self.input['opt_stop'] == 'T':
                try: 
                    parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization
                except OptimizationShouldStop as exc:
                    print(exc.message)
                    break
            elif self.input['opt_stop'] == 'F':
                parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization, still not sure how to make it work
            self.paratrialplot = parameters
            if i >= 5:
                self.step_plotting(num_trial=i)
            # Local evaluation here can be replaced with deployment to external system.
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=self.evaluate_OOP(parameters))
            self.run_time.append(time.time() - self.start)
            #Store current BO trajectory
            if self.input['mult_p'] == 'T':
                self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
                print(type(self.params))
                print(self.params)
                
                self.BO_trace_space_log_x.append([])
                self.BO_trace_space_log_y.append([])
                self.BO_trace_space_log_z.append([])
                self.BO_trace_space_log_x[0].append(self.params[0]['x'])
                self.BO_trace_space_log_y[0].append(self.params[0]['y'])
                self.BO_trace_space_log_z[0].append(self.params[0]['z'])

                if self.input['number_of_ads'] != 1:
                    for i in range(2,self.input['number_of_ads']+1):
                        self.BO_trace_space_log_x.append([])
                        self.BO_trace_space_log_y.append([])
                        self.BO_trace_space_log_z.append([])
                        self.BO_trace_space_log_x[i-1].append(self.params[0]['x'+str(i)])
                        self.BO_trace_space_log_y[i-1].append(self.params[0]['y'+str(i)])
                        self.BO_trace_space_log_z[i-1].append(self.params[0]['z'+str(i)])
                        
            elif self.input['mult_p'] == 'F':
                self.params = self.ax_client.get_best_parameters()[:1][0]
                print(type(self.params))
                print(self.params)
                                    
                self.BO_trace_space_log_x.append([])
                self.BO_trace_space_log_y.append([])
                self.BO_trace_space_log_z.append([])
                self.BO_trace_space_log_x[0].append(self.params['x'])
                self.BO_trace_space_log_y[0].append(self.params['y'])
                self.BO_trace_space_log_z[0].append(self.params['z'])
            
                if self.input['number_of_ads'] != 1:
                    for i in range(2,self.input['number_of_ads']+1):
                        self.BO_trace_space_log_x.append([])
                        self.BO_trace_space_log_y.append([])
                        self.BO_trace_space_log_z.append([])
                        self.BO_trace_space_log_x[i-1].append(self.params['x'+str(i)])
                        self.BO_trace_space_log_y[i-1].append(self.params['y'+str(i)])
                        self.BO_trace_space_log_z[i-1].append(self.params['z'+str(i)])
                    
        if self.input['mult_p'] == 'T':
            BOExperiment.BO_en.append(self.params[1][0]['adsorption_energy'])
        elif self.input['mult_p'] == 'F':
            BOExperiment.BO_en.append(self.ax_client.get_best_parameters()[1][0]['adsorption_energy'])
            
                
        BOExperiment.BO_runtime.append(self.run_time[-1])
        if self.input['mol_soft_constraint'] == 'F':
            if self.input['number_of_ads'] == 1:
                self.plot_acqf()

    def setexp(self):
        self.bulk_z_max = np.max(self.atoms[:-self.n_ads].positions[:, 2]) #modified to account for changes in initial conditions + universal
        self.cell_x_min, self.cell_x_max = float(np.min(self.atoms.cell[:, 0])), float(np.max(self.atoms.cell[:, 0]))
        self.cell_y_min, self.cell_y_max = float(np.min(self.atoms.cell[:, 1])), float(np.max(self.atoms.cell[:, 1]))
        self.z_adsorb_max = self.bulk_z_max + self.input['adsorbant_init_h'] # modified to account for changes in initial conditions
        
        if self.input['mult_p'] == 'T':
            self.objectives={"adsorption_energy": ObjectiveProperties(minimize=True), "dx": ObjectiveProperties(minimize=True), "dy": ObjectiveProperties(minimize=True), "dz": ObjectiveProperties(minimize=True)}
            if self.input['number_of_ads'] != 1:
                for k in range(2,self.input['number_of_ads']+1):
                    self.objectives[f"dx{k}"] = ObjectiveProperties(minimize=True)
                    self.objectives[f"dy{k}"] = ObjectiveProperties(minimize=True)
                    self.objectives[f"dz{k}"] = ObjectiveProperties(minimize=True)
        elif self.input['mult_p'] == 'F':
            self.objectives={"adsorption_energy": ObjectiveProperties(minimize=True)}
            
        if self.input['opt_stop'] == 'T':
            self.stopping_strategy = ImprovementGlobalStoppingStrategy(
            min_trials=5 + 5, window_size=5, improvement_bar=0.01
            )
            self.ax_client = AxClient(generation_strategy=self.gs, global_stopping_strategy=self.stopping_strategy)
        elif self.input['opt_stop'] == 'F':
            self.ax_client = AxClient(generation_strategy=self.gs)
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
        if self.input['number_of_ads'] != 1:
            for i in range(2, self.input['number_of_ads']+1):
                self.parameters.append(
                    {
                        "name": f"x{i}",
                        "type": "range",
                        "bounds": [float(self.cell_x_min), float(self.cell_x_max)/2],
                        "value_type": "float",
                    },
                )
                self.parameters.append(
                    {
                        "name": f"y{i}",
                        "type": "range",
                        "bounds": [float(self.cell_y_min), float(self.cell_y_max)/2],
                        "value_type": "float",
                    },
                )
                self.parameters.append(
                    {
                        "name": f"z{i}",
                        "type": "range",
                        "bounds": [float(self.bulk_z_max), float(self.z_adsorb_max)],
                        "value_type": "float",
                    },)
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
        #Get the trial positions from the model; and set the atoms object to these positions
        x = torch.tensor([parameters.get(f"x"), parameters.get(f"y"), parameters.get(f"z")], device = device)
        xp = []
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                xp.append(torch.tensor([parameters.get(f"x{i}"), parameters.get(f"y{i}"), parameters.get(f"z{i}")], device = device))
        
        new_pos = torch.vstack([torch.zeros((len(self.atoms) - self.input['number_of_ads'], 3), device = device), x])
        
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                new_pos = torch.vstack([new_pos, xp[i-2]])
        
        
        self.atoms.set_positions(new_pos.cpu().numpy(), apply_constraint=True)
        
        #Compute the energy of the system with the current trial positions
        self.energy = torch.tensor(self.atoms.get_potential_energy(), device=device)
            
            
        # Add here Soft constraint for a 2 atom molecule
        # --------------------------------------------- #
        # --------------------------------------------- #
        if self.input['mol_soft_constraint'] == 'T':
            if self.atoms.get_distance(-2, -1) > 1.1:
                print(f'mol. distance : {self.atoms.get_distance(-2, -1)}')
                self.energy = self.energy + (self.atoms.get_distance(-2, -1)-1.1)**self.input['soft_constraint_power']
                print(f'Molecule atom distance: {self.atoms.get_distance(-2, -1)} Angstrom')
                print(f'Soft constraint energy term: {(self.atoms.get_distance(-2, -1)-1.1)**self.input["soft_constraint_power"]}')
        # --------------------------------------------- #
        # --------------------------------------------- #
        # --------------------------------------------- #
        
        
        # Add here Soft constraint for a 3 atom molecule
        # --------------------------------------------- #
        # --------------------------------------------- #
        #if self.input['mol_soft_constraint'] == 'T':
        #    if self.atoms.get_distance(-3, -2) > 2.0:
        #        print(f'mol. distance : {self.atoms.get_distance(-3, -2)}')
        #        self.energy = self.energy + (self.atoms.get_distance(-3, -2)-2.0)**self.input['soft_constraint_power']
        #        print(f'Molecule atom distance: {self.atoms.get_distance(-3, -2)} Angstrom')
        #        print(f'Soft constraint energy term: {(self.atoms.get_distance(-3, -2)-2.0)**self.input["soft_constraint_power"]}')
        #    if self.atoms.get_distance(-3, -1) > 2.0:
        #        print(f'mol. distance : {self.atoms.get_distance(-3, -1)}')
        #        self.energy = self.energy + (self.atoms.get_distance(-3, -1)-2.0)**self.input['soft_constraint_power']
        #        print(f'Molecule atom distance: {self.atoms.get_distance(-3, -1)} Angstrom')
        #        print(f'Soft constraint energy term: {(self.atoms.get_distance(-3, -1)-2.0)**self.input["soft_constraint_power"]}')
        # --------------------------------------------- #
        # --------------------------------------------- #
        # --------------------------------------------- #
        
        #Compute the forces of the system with the current trial positions
        if self.input['mult_p'] == 'T':
            dx,dy,dz = [],[],[]
            dx.append(torch.tensor(self.atoms.get_forces()[-1][0], device = device))
            dy.append(torch.tensor(self.atoms.get_forces()[-1][1], device = device))
            dz.append(torch.tensor(self.atoms.get_forces()[-1][2], device = device))
            #dx = torch.tensor(self.atoms.get_forces()[-1][0], device = device)
            #dy = torch.tensor(self.atoms.get_forces()[-1][1], device = device)
            #dz = torch.tensor(self.atoms.get_forces()[-1][2], device = device)
            if self.input['number_of_ads'] != 1:
                for i in range(2,self.input['number_of_ads']+1):
                    dx.append(torch.tensor(self.atoms.get_forces()[-i][0], device = device))
                    dy.append(torch.tensor(self.atoms.get_forces()[-i][1], device = device))
                    dz.append(torch.tensor(self.atoms.get_forces()[-i][2], device = device))

        #Return the objective values for the trial positions
        # In our case, standard error is 0, since we are computing a synthetic function.
        if self.input['mult_p'] == 'T':
            self.objective_return = {"adsorption_energy": (self.energy, 0.0), "dx": (dx[0], 0.0), "dy": (dy[0], 0.0), "dz": (dz[0], 0.0)}
            if self.input['number_of_ads'] != 1:
                for i in range(2,self.input['number_of_ads']+1):
                    self.objective_return[f"dx{i}"] = (dx[i-1], 0.0)
                    self.objective_return[f"dy{i}"] = (dy[i-1], 0.0)
                    self.objective_return[f"dz{i}"] = (dz[i-1], 0.0)
            return self.objective_return
            
        elif self.input['mult_p'] == 'F':
            return {"adsorption_energy": (self.energy, 0.0)} # We have 0 noise on the target.

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
        
        #Store the optimized BO positions for all the adsorbates, and the energy
        if self.input['mult_p'] == 'T':
            self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
            self.df['opt_bo_x']= self.params[0]['x']
            self.df['opt_bo_y']= self.params[0]['y']
            self.df['opt_bo_z']= self.params[0]['z']
            if self.input['number_of_ads'] != 1:
                for i in range (2,self.input['number_of_ads']+1):
                    self.df['opt_bo_x'+str(i)]= self.params[0]['x'+str(i)]
                    self.df['opt_bo_y'+str(i)]= self.params[0]['y'+str(i)]
                    self.df['opt_bo_z'+str(i)]= self.params[0]['z'+str(i)]
            self.df['opt_bo_energy'] = self.params[1][0]['adsorption_energy']
        elif self.input['mult_p'] == 'F':
            self.params = self.ax_client.get_best_parameters()[:1][0]
            self.df['opt_bo_x']= self.params['x']
            self.df['opt_bo_y']= self.params['y']
            self.df['opt_bo_z']= self.params['z']
            if self.input['number_of_ads'] != 1:
                for i in range (2,self.input['number_of_ads']+1):
                    self.df['opt_bo_x'+str(i)]= self.params['x'+str(i)]
                    self.df['opt_bo_y'+str(i)]= self.params['y'+str(i)]
                    self.df['opt_bo_z'+str(i)]= self.params['z'+str(i)]
            self.df['opt_bo_energy'] = self.ax_client.get_best_parameters()[1][0]['adsorption_energy']
        
        #Store the optimized BFGS energy
        self.df['opt_bfgs_energy'] = self.bfgs_en        
        
        #Save results as dataframe csv file
        self.dfname = f"{self.folder_name}/ase_ads_DF_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.csv"
        
        #Save BO trajectory as dataframe csv file
        self.df_bo_space_trace = pd.DataFrame(list(zip(self.BO_trace_space_log_x[0], self.BO_trace_space_log_y[0], self.BO_trace_space_log_z[0])), columns =['x', 'y', 'z'])
        if self.input['number_of_ads'] != 1:
            for i in range (2,self.input['number_of_ads']+1):
                columns = pd.DataFrame(list(zip(self.BO_trace_space_log_x[i-1], self.BO_trace_space_log_y[i-1], self.BO_trace_space_log_z[i-1])), columns =['x'+str(i), 'y'+str(i), 'z'+str(i)])
                self.df_bo_space_trace = pd.concat([self.df_bo_space_trace, columns], axis=1)
        
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
                    model_kwargs={"seed" : 50},  # Any kwargs you want passed into the model
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
                        "botorch_acqf_class": str_to_class(self.input['bo_acquisition_f']),
                        #"acquisition_options": {"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [1.0]*self.input["number_of_ads"]*3, dtype=torch.double)),"num_fantasies": 5,},
                        #"acquisition_options": {"num_fantasies": 5,},
                                },
                    #model_gen_kwargs={"acquisition_options": {"posterior_transform": ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*self.input["number_of_ads"]*3, dtype=torch.double))},}# Any kwargs you want passed into the model
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

    @classmethod #Probably no longer needed
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
