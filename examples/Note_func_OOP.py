import numpy as np
from ase.build import bulk
import torch
import itertools
import gpytorch
from matplotlib import pyplot as plt
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient, qNoisyExpectedImprovement
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

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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

from botorch.models import SingleTaskGP, ModelListGP, FixedNoiseGP
# Ax wrappers for BoTorch components
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import UpperConfidenceBound

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# ## Create client and initial sampling strategy to warm-up the GP model
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from botorch.models import MultiTaskGP

# Define a class for the experiment global experiment with everything included inside
from ase.eos import calculate_eos
from ase.build import add_adsorbate, fcc111
import random
from ase.constraints import FixAtoms
from ase.visualize import view
import time
import os
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import ase.io
from ase.io import read, write
from ase.collections import g2
from ase.io.animation import write_mp4
import pandas as pd
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.exceptions.core import OptimizationShouldStop
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ase import Atoms
import matplotlib.pyplot as plt
from ax.utils.notebook.plotting import render
from ax.plot.slice import plot_slice
from ax.plot.slice import interact_slice
from ax.plot.slice import _get_slice_predictions
from ax.plot.contour import interact_contour
from ase.visualize.plot import plot_atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.eam import EAM
from ax.core.optimization_config import OptimizationConfig
from ax.core.objective import Objective
from ax.core.metric import Metric
from sklearn.neighbors import KNeighborsRegressor
from ax.plot.contour import _get_contour_predictions


class BOExperiment:
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

    #Easy quick view of the atoms
    def view(self):
        return view(self.atoms, viewer='x3d')

    def best_view(self):
        if self.input['mult_p'] == 'F':
            self.bv_params = self.ax_client.get_best_parameters()[:1][0]
            self.atoms[-1].position[:] = self.bv_params['x'],self.bv_params['y'],self.bv_params['z']
            if self.input['number_of_ads'] != 1:
                for i in range(2,self.input['number_of_ads']+1):
                    self.atoms[-i].position[:] = self.bv_params[f'x{i}'],self.bv_params[f'y{i}'],self.bv_params[f'z{i}']
        elif self.input['mult_p'] == 'T':
            self.bv_params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
            self.atoms[-1].position[:] = self.bv_params[0]['x'],self.bv_params[0]['y'],self.bv_params[0]['z']
            if self.input['number_of_ads'] != 1:
                for i in range(2,self.input['number_of_ads']+1):
                    self.atoms[-i].position[:] = self.bv_params[f'x{i}'],self.bv_params[f'y{i}'],self.bv_params[f'z{i}']
        return view(self.atoms, viewer='x3d')

    #Whole visualization of experiment
    def exp_view(self):
        self.best_view()
        self.BFGS_gif()
        self.BO_gif()
        self.ads_trace_view()
        self.chem_sys_view()
        self.learned_resp_surface()

    def chem_sys_view(self):
        #Plot 1
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        #fig2,ax2 = plt.subplots()
        ax[0].set_title('BO Optimized Adsorption')
        ax[0].set_xlabel("[$\mathrm{\AA}$]")
        ax[0].set_ylabel("[$\mathrm{\AA}$]")
        ax[1].set_xlabel("[$\mathrm{\AA}$]")
        ax[2].set_xlabel("[$\mathrm{\AA}$]")
        ax[3].set_xlabel("[$\mathrm{\AA}$]")
        ax[1].set_title('BO Optimized Adsorption')
        plot_atoms(self.atoms, ax[0], rotation=('90x,45y,0z'), show_unit_cell=True)
        plot_atoms(self.atoms, ax[1], rotation=('0x,0y,0z'))
        #fig.savefig("ase_slab_BO.png")
        ax[2].set_title('BFGS Optimized Adsorption')
        ax[3].set_title('BFGS Optimized Adsorption')
        
        ## Idea to plot several adsorbed atom solutions on the same plot
        from ase import Atoms
        #get all the atom objects
        self.atoms_list = [] #list of atoms objects
        # Plot the last atoms
        for atoms in self.atoms_list:
            # Get the last atom
            self.last_atom = atoms[-1]
            # Create a new Atoms object with only the last atom
            self.last_atom_obj = Atoms([self.last_atom])
            #plot the last atom (adsorbed atom)
            plot_atoms(self.last_atom_obj, ax[0], rotation=('90x,45y,0z'), show_unit_cell=True)
        
        self.atoms_BFGS = self.atoms.copy()
        #self.atoms_BFGS[-1].position[:] = self.BFGS_params[0],self.BFGS_params[1],self.BFGS_params[2]
        self.atoms_BFGS[-1].position[:] = self.BFGSparams[0]
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                self.atoms_BFGS[-i].position[:] = self.BFGSparams[i-1]
        #self.atoms_BFGS[-2].position[:] = self.BFGS_params2[0],self.BFGS_params2[1],self.BFGS_params2[2]
        plot_atoms(self.atoms_BFGS, ax[2], rotation=('90x,45y,0z'), show_unit_cell=True)
        plot_atoms(self.atoms_BFGS, ax[3], rotation=('0x,0y,0z'))
        
        self.filename = f"{self.folder_name}/ase_ads_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.png"
        if self.input["save_fig"] == "T":
            fig.savefig(self.filename)
        
        #Plot 2
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        #fig2,ax2 = plt.subplots()
        ax[0].set_title('BO Optimized Adsorption')
        ax[0].set_xlabel("[$\mathrm{\AA}$]")
        ax[0].set_ylabel("[$\mathrm{\AA}$]")
        ax[1].set_xlabel("[$\mathrm{\AA}$]")
        ax[2].set_xlabel("[$\mathrm{\AA}$]")
        ax[3].set_xlabel("[$\mathrm{\AA}$]")
        ax[1].set_title('BO Optimized Adsorption')
        plot_atoms(self.atoms, ax[0], rotation=('90x,45y,0z'))
        plot_atoms(self.atoms, ax[1], rotation=('0x,0y,0z'))
        #fig.savefig("ase_slab_BO.png")
        ax[2].set_title('BFGS Optimized Adsorption')
        ax[3].set_title('BFGS Optimized Adsorption')
        
        self.atoms_BFGS = self.atoms.copy()
        #self.atoms_BFGS[-1].position[:] = self.BFGS_params[0],self.BFGS_params[1],self.BFGS_params[2]
        self.atoms_BFGS[-1].position[:] = self.BFGSparams[0]
        if self.input['number_of_ads'] != 1:
            for i in range(2,self.input['number_of_ads']+1):
                self.atoms_BFGS[-i].position[:] = self.BFGSparams[i-1]
        #self.atoms_BFGS[-2].position[:] = self.BFGS_params2[0],self.BFGS_params2[1],self.BFGS_params2[2]
        plot_atoms(self.atoms_BFGS, ax[2], rotation=('90x,45y,0z'))
        plot_atoms(self.atoms_BFGS, ax[3], rotation=('0x,0y,0z'))
        
        self.filename = f"{self.folder_name}/ase_ads_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_vacuum_{self.curr_date_time}.png"
        if self.input["save_fig"] == "T":
            fig.savefig(self.filename)

    def learned_resp_surface(self):
        self.model = self.ax_client.generation_strategy.model
        if self.input['mult_p'] == 'T':
            return render(interact_contour(model=self.model, metric_name="adsorption_energy",
                slice_values={'x': self.params[0]['x'], 'y': self.params[0]['y'], 'z': self.params[0]['z']}))
        elif self.input['mult_p'] == 'F':
            return render(interact_contour(model=self.model, metric_name="adsorption_energy",
                slice_values={'x': self.params['x'], 'y': self.params['y'], 'z': self.params['z']}))

    def contour_plot(self, density = 30):
        self.model = self.ax_client.generation_strategy.model
        contour_xy = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "x", y_param_name = "y", density = density, generator_runs_dict = None, slice_values = self.slice_values)
        xy_x = list(contour_xy[3])
        xy_y = list(contour_xy[4])

        xy_e = np.array(contour_xy[1])
        xy_e = xy_e.reshape(len(xy_x),len(xy_y))

        plt.figure()
        plt.contourf(xy_x, xy_y, xy_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['y'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('y [A]')
        plt.title('BO predicted x-y Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/BO_contour_xy.png", dpi=300)
        plt.show()

        contour_xz = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "x", y_param_name = "z", density = density, generator_runs_dict = None, slice_values = self.slice_values)

        xz_x = list(contour_xz[3])
        xz_z = list(contour_xz[4])

        xz_e = np.array(contour_xz[1])
        xz_e = xz_e.reshape(len(xz_x),len(xz_z))

        plt.contourf(xz_x, xz_z, xz_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['z'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted x-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/BO_contour_xz.png", dpi=300)
        plt.show()


        contour_yz = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "y", y_param_name = "z", density = density, generator_runs_dict = None, slice_values = self.slice_values)

        yz_x = list(contour_yz[3])
        yz_z = list(contour_yz[4])

        yz_e = np.array(contour_yz[1])
        yz_e = yz_e.reshape(len(yz_x),len(yz_z))

        plt.contourf(yz_x, yz_z, yz_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['y'], self.slice_values['z'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('y [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted y-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/BO_contour_yz.png", dpi=300)
        plt.show()

    def slice_fit (self, fx=None, fy=None, exp = True):
        if exp == True:    
            self.slice_plot()
            fx,fy,fz = self.slice_values['x'], self.slice_values['y'], self.slice_values['z']
        #self.set_cells()
        #self.pre_ads()
        #self.add_adsorbant()
        self.atoms.calc = EMT()
        
        xmin, xmax = float(0), float(self.cell_x_max/2)
        ymin, ymax = float(0), float(self.cell_y_max/2)
        
        self.x = np.linspace(xmin, xmax, self.density)
        self.y = np.linspace(ymin, ymax, self.density)
        self.z = np.linspace(self.bulk_z_max+1,self.bulk_z_max + self.input['adsorbant_init_h'],self.density)
        
        #Fix y, fit x and z
        self.fyy = [fy]
        self.sl_pos_yfix = list(itertools.product(self.x, self.fyy, self.z))
        self.sl_energies_yfix = []
        for position in self.sl_pos_yfix:
            self.atoms.positions[-1] = position
            energy = self.atoms.get_potential_energy()
            self.sl_energies_yfix.append(energy)
        
        self.sl_E_yfix = np.array(self.sl_energies_yfix).reshape(len(self.x), len(self.z), order = 'F')
        self.sl_knn_yfix = KNeighborsRegressor(n_neighbors=5)
        self.sl_knn_yfix.fit(self.sl_pos_yfix, self.sl_energies_yfix)
        self.sl_E_pred_yfix = self.sl_knn_yfix.predict(self.sl_pos_yfix).reshape(len(self.x), len(self.z), order = 'F')
        
        plt.figure()
        plt.contourf(self.x, self.z, self.sl_E_yfix, 100, label = 'Energy Surface')
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['z'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('z [A]')
        plt.title('Actual x-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/contour_actfit_yfix.png", dpi=300)
        plt.show()
        
        #Fix x, fit y and z
        self.fxx = [fx]
        self.sl_pos_xfix = list(itertools.product(self.fxx, self.y, self.z))
        self.sl_energies_xfix = []
        for position in self.sl_pos_xfix:
            self.atoms.positions[-1] = position
            energy = self.atoms.get_potential_energy()
            self.sl_energies_xfix.append(energy)
        
        self.sl_E_xfix = np.array(self.sl_energies_xfix).reshape(len(self.y), len(self.z), order = 'F')
        self.sl_knn_xfix = KNeighborsRegressor(n_neighbors=5)
        self.sl_knn_xfix.fit(self.sl_pos_xfix, self.sl_energies_xfix)
        self.sl_E_pred_xfix = self.sl_knn_xfix.predict(self.sl_pos_xfix).reshape(len(self.y), len(self.z), order = 'F')
        
        plt.figure()
        plt.contourf(self.y, self.z, self.sl_E_xfix, 100, label = 'Energy Surface')
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['y'], self.slice_values['z'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('y [A]')
        plt.ylabel('z [A]')
        plt.title('Actual y-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/contour_actfit_xfix.png", dpi=300)
        plt.show()

        #Fix z, fit x and y
        self.fzz = [fz]
        self.sl_pos_zfix = list(itertools.product(self.x, self.y, self.fzz))
        self.sl_energies_zfix = []
        for position in self.sl_pos_zfix:
            self.atoms.positions[-1] = position
            energy = self.atoms.get_potential_energy()
            self.sl_energies_zfix.append(energy)
        
        self.sl_E_zfix = np.array(self.sl_energies_zfix).reshape(len(self.x), len(self.y), order = 'F')
        self.sl_knn_zfix = KNeighborsRegressor(n_neighbors=5)
        self.sl_knn_zfix.fit(self.sl_pos_zfix, self.sl_energies_zfix)
        self.sl_E_pred_zfix = self.sl_knn_zfix.predict(self.sl_pos_zfix).reshape(len(self.x), len(self.y), order = 'F')
        
        plt.figure()
        plt.contourf(self.x, self.y, self.sl_E_zfix, 100, label = 'Energy Surface')
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['y'], color = "red", label = "BO optimum", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('y [A]')
        plt.title('Actual x-y Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name}/contour_actfit_zfix.png", dpi=300)
        plt.show()

    def slice_plot(self):
        self.model = self.ax_client.generation_strategy.model
        if self.input['mult_p'] == 'T':
            self.params = self.ax_client.get_pareto_optimal_parameters()[next(iter(self.ax_client.get_pareto_optimal_parameters()))]
            self.slice_values = {'x': self.params[0]['x'], 'y': self.params[0]['y'], 'z': self.params[0]['z']}
        elif self.input['mult_p'] == 'F':
            self.params = self.ax_client.get_best_parameters()[:1][0]
            self.slice_values = {'x': self.params['x'], 'y': self.params['y'], 'z': self.params['z']}
        print(type(self.params))
        print(self.params)
        #render(interact_slice(model = self.model, slice_values = self.slice_values))
        
        self.slice_x = _get_slice_predictions(model = self.model, param_name="x", metric_name="adsorption_energy", slice_values=self.slice_values)
        self.slice_plot_en_x = self.slice_x[2]
        self.slice_plot_x = self.slice_x[4]
        self.slice_y = _get_slice_predictions(model = self.model, param_name="y", metric_name="adsorption_energy", slice_values=self.slice_values)
        self.slice_plot_en_y = self.slice_y[2]
        self.slice_plot_y = self.slice_y[4]
        
        #Not used yet
        #self.slice_pos_list = list(_get_slice_predictions(model = self.model, param_name="x", metric_name="adsorption_energy", slice_values=self.slice_values)[1].values())
        #self.x_slice = []
        #self.y_slice = []
        #self.enx_slice = []
        #self.eny_slice = []
        #for k in range(len(_get_slice_predictions(model = self.model, param_name="x", metric_name="adsorption_energy", slice_values=self.slice_values)[1])):
        #    self.x_slice.append(self.slice_pos_list[k]['x'])
        #    self.y_slice.append(self.slice_pos_list[k]['y'])
        #    self.enx_slice.append(list(_get_slice_predictions(model = self.model, param_name="x", metric_name="adsorption_energy", slice_values=self.slice_values)[3])[k]['mean'])
        #    self.eny_slice.append(list(_get_slice_predictions(model = self.model, param_name="y", metric_name="adsorption_energy", slice_values=self.slice_values)[3])[k]['mean'])

    def slice_plot_resp_surface(self, fx=None, fy=None, exp = True, density = 30):
        self.density = density
        self.exp = exp
        self.slice_fit(exp = self.exp)
        #if exp == True:    
        #    fx,fy = self.slice_values['x'], self.slice_values['y']
        #self.filename = 'knn_model.sav'
        #Load the model from a file
        #loaded_model = pickle.load(open(self.filename, 'rb'))
        
        self.jx = float(self.cell_x_max/2)
        self.jy = float(self.cell_y_max/2)
        x_test = np.linspace(0, self.jx, 30)
        y_test = np.linspace(0, self.jy, 30)
        #positions_test = list(itertools.product(x_test, y_test))
        #Predict energies using old model
        #E_test = loaded_model.predict(positions_test).reshape(len(x_test),len(y_test), order = 'F')
        
        #Predict energies using updated model
        fz = self.slice_values['z']
        z_test = [fz]
        
        #x fixed
        self.sl_E_test_xfix = self.sl_knn_xfix.predict(list(itertools.product(self.fxx, y_test, z_test))).reshape(len(y_test),len(z_test), order = 'F')
        plt.plot(y_test, self.sl_E_test_xfix, color = "darkblue", label = "Actual Energy Surface")
        plt.plot(self.slice_plot_y, self.slice_plot_en_y, color = "red", label = "BO model")
        plt.ylabel('Energy')
        plt.xlabel('y [A]')
        plt.xlim(0, self.jy)
        plt.grid()
        plt.legend()
        plt.title('BO Predicted vs Actual Energy Surface (x fixed at optimum)')
        plt.savefig(f"{self.folder_name}/slice_predfit_xfix.png", dpi=300)
        plt.show()

        #y fixed 
        self.sl_E_test_yfix = self.sl_knn_yfix.predict(list(itertools.product(x_test,self.fyy, z_test))).reshape(len(x_test),len(z_test), order = 'F')
        plt.plot(x_test, self.sl_E_test_yfix, color = "darkblue", label = "Actual Energy Surface")
        plt.plot(self.slice_plot_x, self.slice_plot_en_x, color = "red", label = "BO model")
        plt.ylabel('Energy')
        plt.xlabel('x [A]')
        plt.xlim(0, self.jx)
        plt.grid()
        plt.legend()
        plt.title('BO Predicted vs Actual Energy Surface (y fixed at optimum)')
        plt.savefig(f"{self.folder_name}/slice_predfit_yfix.png", dpi=300)
        plt.show()

        ## Plot for x,z fixed
        #self.jxx = fx/self.jx *30
        #print(self.jxx)
        #plt.plot(y_test, E_test[:,int(self.jxx)], color = "darkred") #Index will define what value of x is fixed (index+1/100 * max x)
        #plt.plot(self.slice_plot_y, self.slice_plot_en_y, color = "darkblue")
        ##plt.scatter(self.y_slice, self.eny_slice, color = "darkblue")
        #plt.ylabel('Energy')
        #plt.xlabel('y [A]')
        #plt.grid()
        #plt.show()
        #print('fixed x = ', fx)
        #
        #
        ##plot for y,z fixed
        #self.jyy = fy/self.jy *30
        #plt.plot(x_test, E_test[int(self.jyy)], color = "darkred") #Index will define what value of y is fixed (index+1/100 * max y)
        #plt.plot(self.slice_plot_x, self.slice_plot_en_x, color = "darkblue")
        ##plt.scatter(self.x_slice, self.enx_slice, color = "darkblue")
        #plt.ylabel('Energy')
        #plt.xlabel('x [A]')
        #plt.grid()
        #plt.show()
        #print('fixed y = ', fy)

    def ads_trace_view(self):
        # Plot the optimization trace vs steps
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].set_title('BO Optimized Adsorption vs steps')
        ax[0].set_xlabel("Optimization step")
        ax[0].set_ylabel("Current optimum")
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].grid(True, linestyle='--', color='0.7', zorder=-1, linewidth=1, alpha=0.5)
        # Add horizontal line at x = gs_init_steps to indicate the end of the initialization trials.
        ax[0].axvline(x=self.input['gs_init_steps'], color='k', linestyle='--', linewidth=2, alpha=0.5, label='End of initialization trials')
        
        #bfgs
        x_bfgs = range(len(self.df_bfgs))
        y_bfgs = self.df_bfgs['Energy']
        ax[0].plot(x_bfgs, y_bfgs, label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        
        #BO
        trace = self.df['bo_trace']
        x = range(len(trace))
        ax[0].plot(x, trace, label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        
        # Plot the optimization trace vs time
        ax[1].set_title('BO Optimized Adsorption vs time')
        ax[1].set_xlabel("Optimization time (s)")
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].grid(True, linestyle='--', color='0.7', zorder=-1, linewidth=1, alpha=0.5)
        #BFGS
        xt_bfgs = self.df_bfgs['Time']
        ax[1].plot(xt_bfgs, y_bfgs, label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        #BO
        xt_BO = self.df['run_time']
        ax[1].axvline(x=self.df['run_time'][self.input['gs_init_steps']-1], color='k', linestyle='--', linewidth=2, alpha=0.5, label='End of initialization trials')
        ax[1].plot(xt_BO, trace, label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        
        plt.legend()
        ax[0].legend()
        if self.input["save_fig"] == "T":
            fig.savefig(f"{self.folder_name}/ase_ads_Opt_trace_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.png")        

    def BO_gif(self):
        self.atoms_copy = self.atoms.copy()
        self.traj = Trajectory(f'{self.folder_name}/BO.traj', 'w', self.atoms)
        self.traj_rot = Trajectory(f'{self.folder_name}/BO_rot.traj', 'w', self.atoms_copy)
        
        self.traj_trial = Trajectory(f'{self.folder_name}/trial_atom.traj', 'w')
        self.traj_trial_rot = Trajectory(f'{self.folder_name}/trial_atom_rot.traj', 'w')
        self.traj_ptrial_blender = Trajectory(f'{self.folder_name}/trial_atom_blender.traj', 'w')
        
        self.traj_trial2 = Trajectory(f'{self.folder_name}/trial_atom2.traj', 'w')
        self.traj_trial_rot2 = Trajectory(f'{self.folder_name}/trial_atom_rot2.traj', 'w')
        self.traj_ptrial_blender2 = Trajectory(f'{self.folder_name}/trial_atom_blender2.traj', 'w')
        
        #Transform df_bo_space_trace to ASE trajectory
        for i in range(len(self.df_bo_space_trace)):
            self.atoms[-1].position[:] = self.df_bo_space_trace['x'][i],self.df_bo_space_trace['y'][i],self.df_bo_space_trace['z'][i]
            if self.input['number_of_ads'] != 1:
                for j in range(2,self.input['number_of_ads']+1):
                    self.atoms[-j].position[:] = self.df_bo_space_trace[f'x{j}'][i],self.df_bo_space_trace[f'y{j}'][i],self.df_bo_space_trace[f'z{j}'][i]
            #self.atoms[-2].position[:] = self.df_bo_space_trace['x2'][i],self.df_bo_space_trace['y2'][i],self.df_bo_space_trace['z2'][i]
            self.traj.write(self.atoms)
            self.traj_ptrial_blender.write(self.atoms)
            
            self.atoms_copy = self.atoms.copy()
            self.atoms_copy.translate([0, 0, 0])
            self.atoms_copy.rotate(90, 'x')
            self.atoms_copy.rotate(45, 'y')
            self.traj_rot.write(self.atoms_copy)
            
            self.trial_atom = Atoms('B', positions=[[self.df['x'][i], self.df['y'][i], self.df['z'][i]]])
            self.trial_atom.set_cell(self.atoms.get_cell())
            self.trial_atom.set_pbc(self.atoms.get_pbc())
            self.traj_trial.write(self.trial_atom)
            self.traj_ptrial_blender.write(self.trial_atom)
            
            self.trial_copy = self.trial_atom.copy()
            self.trial_copy.rotate(90, 'x')
            self.trial_copy.rotate(45, 'y')
            self.traj_trial_rot.write(self.trial_copy)
            if self.input['number_of_ads'] != 1:
                self.trial_atom2 = Atoms('B', positions=[[self.df['x2'][i], self.df['y2'][i], self.df['z2'][i]]])
                self.trial_atom2.set_cell(self.atoms.get_cell())
                self.trial_atom2.set_pbc(self.atoms.get_pbc())
                self.traj_trial2.write(self.trial_atom2)
                self.traj_ptrial_blender2.write(self.trial_atom2)
                
                self.trial_copy2 = self.trial_atom2.copy()
                self.trial_copy2.rotate(90, 'x')
                self.trial_copy2.rotate(45, 'y')
                self.traj_trial_rot2.write(self.trial_copy2)
            #self.trial_atom2 = Atoms('B', positions=[[self.df['x2'][i], self.df['y2'][i], self.df['z2'][i]]])
            #self.trial_atom2.set_cell(self.atoms.get_cell())
            #self.trial_atom2.set_pbc(self.atoms.get_pbc())
            #self.traj_trial2.write(self.trial_atom2)
            #self.traj_ptrial_blender2.write(self.trial_atom2)
            #
            #self.trial_copy2 = self.trial_atom2.copy()
            #self.trial_copy2.rotate(90, 'x')
            #self.trial_copy2.rotate(45, 'y')
            #self.traj_trial_rot2.write(self.trial_copy2)
        
        self.BO_atoms_list = list(Trajectory(f'{self.folder_name}/BO.traj'))
        self.BO_atoms_list_rot = list(Trajectory(f'{self.folder_name}/BO_rot.traj'))
        
        self.traj_trial_list = list(Trajectory(f'{self.folder_name}/trial_atom.traj'))
        self.traj_trial_list_rot = list(Trajectory(f'{self.folder_name}/trial_atom_rot.traj'))
        if self.input['number_of_ads'] != 1:
            self.traj_trial_list2 = list(Trajectory(f'{self.folder_name}/trial_atom2.traj'))
            self.traj_trial_list_rot2 = list(Trajectory(f'{self.folder_name}/trial_atom_rot2.traj'))
        
        self.combined_atoms_list = []
        if self.input['number_of_ads'] == 1:
            for atoms, atoms_rot, trial_a, trial_a_rot in zip(self.BO_atoms_list, self.BO_atoms_list_rot, self.traj_trial_list, self.traj_trial_list_rot):
                # Translate atoms_test to avoid overlap
                atoms_rot.translate([0, 0, 0])  # Adjust the translation vector as needed
                self.combined_atoms_list.append(atoms + atoms_rot + trial_a + trial_a_rot)
        elif self.input['number_of_ads'] != 1:
            for atoms, atoms_rot, trial_a, trial_a_rot, trial2_a, trial2_a_rot in zip(self.BO_atoms_list, self.BO_atoms_list_rot, self.traj_trial_list, self.traj_trial_list_rot, self.traj_trial_list2, self.traj_trial_list_rot2):
                # Translate atoms_test to avoid overlap
                atoms_rot.translate([0, 0, 0])  # Adjust the translation vector as needed
                self.combined_atoms_list.append(atoms + atoms_rot + trial_a + trial_a_rot + trial2_a + trial2_a_rot)
        
        if self.input["save_fig"] == "T":
            ase.io.write(f'{self.folder_name}/BO_space_trace.gif', self.combined_atoms_list, interval=800)
            #write_mp4(f'{self.folder_name}/BO_space_trace.mp4', self.combined_atoms_list, interval=800)

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

    def BFGS_gif(self):
        self.BFGS_traj = Trajectory(f'{self.folder_name}/BFGS.traj')
        self.bfgs_list = list(Trajectory(f'{self.folder_name}/BFGS.traj'))
        
        self.rot_bfgs_list = list(Trajectory(f'{self.folder_name}/BFGS.traj'))
        for atoms_rot in self.rot_bfgs_list:
            atoms_rot.rotate(90, 'x')
            atoms_rot.rotate(45, 'y') 
        
        self.bfgs_combined_atoms_list = []
        
        for atoms_bfgs, atoms_rot in zip(self.bfgs_list, self.rot_bfgs_list):
            # Translate atoms_rot to avoid overlap on final plot
            atoms_rot.translate([0, 0, 0])  # Adjust the translation vector as needed
            self.bfgs_combined_atoms_list.append(atoms_bfgs + atoms_rot)
        
        if self.input["save_fig"] == "T":
            ase.io.write(f'{self.folder_name}/BFGS_traj.gif', self.bfgs_combined_atoms_list, interval=100) #Could save as video as well
            #ase.io.animation.write_mp4(f'{self.folder_name}/BFGS_traj', self.bfgs_combined_atoms_list, interval=100)

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

    @classmethod
    def add_exp(cls):
        cls.number_of_exp += 1

    def conv_study(self):
        # To add later
        pass

    def response_surface_init(self):
        self.set_cells()
        self.pre_ads()
        self.add_adsorbant()
        self.atoms.calc = EMT()
        self.filtered_atoms_1 = self.atoms[self.atoms.get_tags() == 1]
        self.filtered_atoms_2 = self.atoms[self.atoms.get_tags() == 2]
        self.filtered_atoms_3 = self.atoms[self.atoms.get_tags() == 3]
        xmin, xmax = float(np.min(self.atoms.positions[:, 0])), float(np.max(self.atoms.positions[:, 0]))
        ymin, ymax = float(np.min(self.atoms.positions[:, 1])), float(np.max(self.atoms.positions[:, 1]))
        self.x = np.linspace(xmin, xmax, 100)
        self.y = np.linspace(ymin, ymax, 100)
        self.z = [14.65557600]
        #Make a list of positions x,y,z
        positions = list(itertools.product(self.x, self.y, self.z))
        #Calculate the energy of each position
        self.energies = []
        for position in positions:
            self.atoms.positions[-1] = position
            energy = self.atoms.get_potential_energy()
            self.energies.append(energy)
        #save self.energies to a csv file
        np.savetxt('response_surface2.csv', self.energies, delimiter=',')
        np.savetxt('surface_positions2.csv', positions, delimiter=',')
        np.savetxt('x_response_surface2.csv', self.x, delimiter=',')
        np.savetxt('y_response_surface2.csv', self.y, delimiter=',')

    def Response_surface(self):
        self.response_surface_init()
        #Reshape E and check
        E = np.array(self.energies).reshape(len(self.x),len(self.y), order = 'F') #Issue was the reshaping was accounting the wrong order 
        # Save the E array to a csv file
        #was using a C-like index order, but the correct one is Fortran-like index order
        # 2D plot
        plt.figure()
        plt.contourf(self.x, self.y, E, levels=100, label = 'Energy')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        # Make it so x and y are on the same scale
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot the filtered atom positions
        plt.scatter(self.filtered_atoms_1.positions[:, 0], self.filtered_atoms_1.positions[:, 1], c='red', label = 'Layer 1')
        plt.scatter(self.filtered_atoms_2.positions[:, 0], self.filtered_atoms_2.positions[:, 1], c='blue', label = 'Layer 2')
        plt.scatter(self.filtered_atoms_3.positions[:, 0], self.filtered_atoms_3.positions[:, 1], c='green', label = 'Layer 3')
        plt.legend()
        plt.show()

    def two_atom_response_surface(self):
        print("Don't forget that input must be 2 adsorbed atoms for this to work!")
        self.set_cells()
        self.pre_ads()
        self.add_adsorbant()
        self.atoms.calc = EMT()
        self.atoms.positions[-2] = [0, 0, self.z]
        self.filtered_atoms_1 = self.atoms[self.atoms.get_tags() == 1]
        self.filtered_atoms_2 = self.atoms[self.atoms.get_tags() == 2]
        self.filtered_atoms_3 = self.atoms[self.atoms.get_tags() == 3]
        xmin, xmax = float(np.min(self.atoms.positions[:, 0])), float(np.max(self.atoms.positions[:, 0]))
        ymin, ymax = float(np.min(self.atoms.positions[:, 1])), float(np.max(self.atoms.positions[:, 1]))
        self.x = np.linspace(xmin, xmax, 100)
        self.y = np.linspace(ymin, ymax, 100)
        self.z = [14.65557600]
        #Make a list of positions x,y,z
        positions = list(itertools.product(self.x, self.y, self.z))
        #Calculate the energy of each position
        self.energies = []
        for position in positions:
            self.atoms.positions[-1] = position
            energy = self.atoms.get_potential_energy()
            self.energies.append(energy)
        #Reshape E and check
        E = np.array(self.energies).reshape(len(self.x),len(self.y), order = 'F') #Issue was the reshaping was accounting the wrong order 
        #was using a C-like index order, but the correct one is Fortran-like index order
        # 2D plot
        plt.figure()
        plt.contourf(self.x, self.y, E, levels=100, label = 'Energy')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        # Make it so x and y are on the same scale
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot the filtered atom positions
        plt.scatter(self.filtered_atoms_1.positions[:, 0], self.filtered_atoms_1.positions[:, 1], c='red', label = 'Layer 1')
        plt.scatter(self.filtered_atoms_2.positions[:, 0], self.filtered_atoms_2.positions[:, 1], c='blue', label = 'Layer 2')
        plt.scatter(self.filtered_atoms_3.positions[:, 0], self.filtered_atoms_3.positions[:, 1], c='green', label = 'Layer 3')
        plt.legend()
        plt.show()

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
        
        if self.input['opt_stop'] == 'T':
            #run this if we want to stop the optimization after a threshold
            for i in range(self.N_BO_steps):
                try: 
                    parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization
                except OptimizationShouldStop as exc:
                    print(exc.message)
                    break
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
                parameters, trial_index = self.ax_client.get_next_trial() #Use .get_next_trials() for parallel optimization, still not sure how to make it work
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

    #Load previous experiment
    def load_exp(self, exp_name):
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

    #Save experiment and Ax client as JSON file 
    def save_exp_json(self):
        self.ax_client.save_to_json_file(filepath= f"{self.folder_name}/ase_ads_DF_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.json")

    #Save experiment as csv
    def save_exp_csv(self):
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
                        #"acquisition_options": {"torch_device": device},
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

    @classmethod
    def record_data(cls):
        #To do later
        pass