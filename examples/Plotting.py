import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
import torch
import itertools
from matplotlib import pyplot as plt

# Check if CUDA is available and set PyTorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Define a class for the experiment global experiment with everything included inside
from ase.visualize import view
from ase.io.trajectory import Trajectory
import ase.io
from ase import Atoms
import matplotlib.pyplot as plt
from ax.utils.notebook.plotting import render
from ax.plot.slice import _get_slice_predictions
from ax.plot.contour import interact_contour
from ase.visualize.plot import plot_atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.eam import EAM
from sklearn.neighbors import KNeighborsRegressor
from ax.plot.contour import _get_contour_predictions
from ax.core import arm, observation
import imageio
import os

class plotter:
    def step_plotting(self, num_trial = 0): #record contour plots after every BO step
        self.slice_plot()
        self.contour_plot(num_trial=num_trial)
        self.slice_plot_resp_surface(num_trial=num_trial, fit = False)

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

    def plot_acqf(self, num_trial = 0):
        #create a folder to save the plots
        self.folder_name_acqf = f"{self.folder_name}/acqf_images"
        if not os.path.exists(self.folder_name_acqf):
            os.makedirs(self.folder_name_acqf)
        #plot the acquisition function
        num_trial = 6
        for q in self.BO_models:
            #x plot
            self.acqf_eval_x = np.zeros(len(self.slice_x[4]))
            for p in range(len(self.slice_x[4])):
                a = arm.Arm({'x': self.slice_x[4][p], 'y': self.ax_client.get_trials_data_frame()['y'][num_trial], 'z':  self.ax_client.get_trials_data_frame()['z'][num_trial]})
                b = observation.ObservationFeatures.from_arm(a)
                self.acqf_eval_x[p] = q.evaluate_acquisition_function([b])[0]
            #y plot
            self.acqf_eval_y = np.zeros(len(self.slice_y[4]))
            for p in range(len(self.slice_y[4])):
                a = arm.Arm({'x': self.ax_client.get_trials_data_frame()['x'][num_trial], 'y': self.slice_y[4][p], 'z': self.ax_client.get_trials_data_frame()['z'][num_trial]})
                b = observation.ObservationFeatures.from_arm(a)
                self.acqf_eval_y[p] = q.evaluate_acquisition_function([b])[0]
            #z plot
            self.acqf_eval_z = np.zeros(len(self.slice_z[4]))
            for p in range(len(self.slice_z[4])):
                a = arm.Arm({'x': self.ax_client.get_trials_data_frame()['x'][num_trial], 'y': self.ax_client.get_trials_data_frame()['y'][num_trial], 'z': self.slice_z[4][p]})
                b = observation.ObservationFeatures.from_arm(a)
                self.acqf_eval_z[p] = q.evaluate_acquisition_function([b])[0]

            #y,z fixed
            plt.plot(self.slice_plot_x, self.acqf_eval_x, color = "red", label = "BO model")
            plt.ylabel('Criterion Value')
            plt.xlabel('x [A]')
            plt.xlim(0, self.jx)
            plt.grid()
            plt.legend()
            plt.title(f'BO Acquisition criterion vs x, y = {self.ax_client.get_trials_data_frame()["y"][num_trial]:.3f} z = {self.ax_client.get_trials_data_frame()["z"][num_trial]:.3f}')
            plt.savefig(f"{self.folder_name_acqf}/yfix_slice_acqf_{num_trial}.png", dpi=300)
            plt.show()

            #x,z fixed
            plt.plot(self.slice_plot_y, self.acqf_eval_y, color = "red", label = "BO model")
            plt.ylabel('Criterion Value')
            plt.xlabel('y [A]')
            plt.xlim(0, self.jy)
            plt.grid()
            plt.legend()
            plt.title(f'BO Acquisition criterion vs y, x = {self.ax_client.get_trials_data_frame()["x"][num_trial]:.3f} z = {self.ax_client.get_trials_data_frame()["z"][num_trial]:.3f}')
            plt.savefig(f"{self.folder_name_acqf}/xfix_slice_acqf_{num_trial}.png", dpi=300)
            plt.show()

            #x,y fixed
            plt.plot(self.slice_plot_z, self.acqf_eval_z, color = "red", label = "BO model")
            plt.ylabel('Criterion Value')
            plt.xlabel('z [A]')
            plt.grid()
            plt.legend()
            plt.title(f'BO Acquisition criterion vs z, x = {self.ax_client.get_trials_data_frame()["x"][num_trial]:.3f} y = {self.ax_client.get_trials_data_frame()["y"][num_trial]:.3f}')
            plt.savefig(f"{self.folder_name_acqf}/zfix_slice_acqf_{num_trial}.png", dpi=300)
            plt.show()
            
            num_trial += 1
        #for q in self.BO_models: # Attemps of data for contour plot of the acquisition function but very expensive (more than 1min per trial)
        #    #x plot
        #    self.acqf_eval_xy = []
        #    for p in range(len(self.slice_x[4])):
        #        for s in range(len(self.slice_y[4])):
        #            a = arm.Arm({'x': self.slice_x[4][p], 'y': self.slice_y[4][s], 'z': self.slice_values['z']})
        #            b = observation.ObservationFeatures.from_arm(a)
        #            self.acqf_eval_xy.append(q.evaluate_acquisition_function([b])[0])
        #    #y plot
        #    self.acqf_eval_yz = []
        #    for p in range(len(self.slice_y[4])):
        #        for s in range(len(self.slice_z[4])):
        #            a = arm.Arm({'x': self.slice_values['x'], 'y': self.slice_y[4][p], 'z': self.slice_z[4][s]})
        #            b = observation.ObservationFeatures.from_arm(a)
        #            self.acqf_eval_yz.append(q.evaluate_acquisition_function([b])[0])
        #    #z plot
        #    self.acqf_eval_zx = []
        #    for p in range(len(self.slice_z[4])):
        #        for s in range(len(self.slice_x[4])):
        #            a = arm.Arm({'x': self.slice_x[4][s], 'y': self.slice_values['y'], 'z': self.slice_z[4][p]})
        #            b = observation.ObservationFeatures.from_arm(a)
        #            self.acqf_eval_zx.append(q.evaluate_acquisition_function([b])[0])
        #
        #print(self.acqf_eval_xy)
        #print(len(self.acqf_eval_xy))
        #print(self.slice_y[4][1])

    def contour_plot(self, density = 30, num_trial = 0):
        #create a folder to save the plots
        self.folder_name_contour = f"{self.folder_name}/contour_images"
        if not os.path.exists(self.folder_name_contour):
            os.makedirs(self.folder_name_contour)
        #plot the contour plots
        self.model = self.ax_client.generation_strategy.model
        self.trials_plot = self.ax_client.get_trials_data_frame()
        contour_xy = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "x", y_param_name = "y", density = density, generator_runs_dict = None, slice_values = self.slice_values)
        xy_x = list(contour_xy[3])
        xy_y = list(contour_xy[4])

        xy_e = np.array(contour_xy[1])
        xy_e = xy_e.reshape(len(xy_x),len(xy_y))
        xy_se = np.array(contour_xy[2])
        xy_se = xy_se.reshape(len(xy_x),len(xy_y))

        plt.contourf(xy_x, xy_y, xy_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['y'], color = "red", label = f"BO optimum, z = {self.slice_values['z']:.3f}", marker="D")
        plt.scatter(self.trials_plot['x'], self.trials_plot['y'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['x'], self.paratrialplot['y'], color = "green", label = f"Next trial, z = {self.paratrialplot['z']:.3f}", marker="D",linewidths=2)
        plt.axhline(self.paratrialplot['y'], color = "red", linestyle = "--", alpha = 0.4, label = "x-Acquisition function plot")
        plt.axvline(self.paratrialplot['x'], color = "red", linestyle = "--", alpha = 0.4, label = "y-Acquisition function plot")
        plt.xlabel('x [A]')
        plt.ylabel('y [A]')
        plt.title('BO predicted x-y Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/xy_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

        plt.figure()
        plt.contourf(xy_x, xy_y, xy_se, 100, cmap='plasma')    
        plt.colorbar(label = 'Standard Error')
        plt.scatter(self.slice_values['x'], self.slice_values['y'], color = "red", label = f"BO optimum, z = {self.slice_values['z']:.3f}", marker="D")
        plt.scatter(self.trials_plot['x'], self.trials_plot['y'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['x'], self.paratrialplot['y'], color = "green", label = f"Next trial, z = {self.paratrialplot['z']:.3f}", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('y [A]')
        plt.title('BO predicted x-y Standard Error surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/xy_se_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

        ## 3D plot
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(xy_x, xy_y, xy_e)
        #ax.set_xlabel('x [A]')
        #ax.set_ylabel('y [A]')
        #ax.set_zlabel('Energy')
        #ax.set_title('BO predicted x-y Energy surface')
        ##Set the view angle
        #ax.view_init(15, 15)
        #plt.savefig(f"{self.folder_name}/BO_contour_xy_3D_{num_trial}.png", dpi=300)
        #plt.show()

        contour_xz = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "x", y_param_name = "z", density = density, generator_runs_dict = None, slice_values = self.slice_values)

        xz_x = list(contour_xz[3])
        xz_z = list(contour_xz[4])

        xz_e = np.array(contour_xz[1])
        xz_e = xz_e.reshape(len(xz_x),len(xz_z))
        xz_se = np.array(contour_xz[2])
        xz_se = xz_se.reshape(len(xz_x),len(xz_z))

        plt.contourf(xz_x, xz_z, xz_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['x'], self.slice_values['z'], color = "red", label = f"BO optimum, y = {self.slice_values['y']:.3f}", marker="D")
        plt.scatter(self.trials_plot['x'], self.trials_plot['z'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['x'], self.paratrialplot['z'], color = "green", label = f"Next trial, y = {self.paratrialplot['y']:.3f}", marker="D")
        plt.axhline(self.paratrialplot['z'], color = "red", linestyle = "--", alpha = 0.4, label = "x-Acquisition function plot")
        plt.axvline(self.paratrialplot['x'], color = "red", linestyle = "--", alpha = 0.4, label = "z-Acquisition function plot")
        plt.xlabel('x [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted x-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/xz_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

        plt.figure()
        plt.contourf(xz_x, xz_z, xz_se, 100, cmap='plasma')
        plt.colorbar(label = 'Standard Error')
        plt.scatter(self.slice_values['x'], self.slice_values['z'], color = "red", label = f"BO optimum, y = {self.slice_values['y']:.3f}", marker="D")
        plt.scatter(self.trials_plot['x'], self.trials_plot['z'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['x'], self.paratrialplot['z'], color = "green", label = f"Next trial, y = {self.paratrialplot['y']:.3f}", marker="D")
        plt.xlabel('x [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted x-z Standard Error surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/xz_se_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

        contour_yz = _get_contour_predictions(model = self.model, metric = "adsorption_energy", x_param_name = "y", y_param_name = "z", density = density, generator_runs_dict = None, slice_values = self.slice_values)

        yz_x = list(contour_yz[3])
        yz_z = list(contour_yz[4])

        yz_e = np.array(contour_yz[1])
        yz_e = yz_e.reshape(len(yz_x),len(yz_z))
        yz_se = np.array(contour_yz[2])
        yz_se = yz_se.reshape(len(yz_x),len(yz_z))

        plt.contourf(yz_x, yz_z, yz_e, 100)
        plt.colorbar(label = 'Energy Surface')
        plt.scatter(self.slice_values['y'], self.slice_values['z'], color = "red", label = f"BO optimum, x = {self.slice_values['x']:.3f}", marker="D")
        plt.scatter(self.trials_plot['y'], self.trials_plot['z'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['y'], self.paratrialplot['z'], color = "green", label = f"Next trial, x = {self.paratrialplot['x']:.3f}", marker="D")
        plt.axhline(self.paratrialplot['z'], color = "red", linestyle = "--", alpha = 0.4, label = "y-Acquisition function plot")
        plt.axvline(self.paratrialplot['y'], color = "red", linestyle = "--", alpha = 0.4, label = "z-Acquisition function plot")
        plt.xlabel('y [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted y-z Energy surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/yz_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

        plt.figure()
        plt.contourf(yz_x, yz_z, yz_se, 100, cmap='plasma')
        plt.colorbar(label = 'Standard Error')
        plt.scatter(self.slice_values['y'], self.slice_values['z'], color = "red", label = f"BO optimum, x = {self.slice_values['x']:.3f}", marker="D")
        plt.scatter(self.trials_plot['y'], self.trials_plot['z'], color = "black", label = "Trials", marker="x")
        plt.scatter(self.paratrialplot['y'], self.paratrialplot['z'], color = "green", label = f"Next trial, x = {self.paratrialplot['x']:.3f}", marker="D")
        plt.xlabel('y [A]')
        plt.ylabel('z [A]')
        plt.title('BO predicted y-z Standard Error surface')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig(f"{self.folder_name_contour}/yz_se_BO_contour_{num_trial}.png", dpi=300)
        plt.show()

    def slice_fit (self, fx=None, fy=None, exp = True, plot = False):
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
        
        if plot == True:
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
        self.slice_plot_x_se = self.slice_x[9]
        self.slice_y = _get_slice_predictions(model = self.model, param_name="y", metric_name="adsorption_energy", slice_values=self.slice_values)
        self.slice_plot_en_y = self.slice_y[2]
        self.slice_plot_y = self.slice_y[4]
        self.slice_plot_y_se = self.slice_y[9]
        self.slice_z = _get_slice_predictions(model = self.model, param_name="z", metric_name="adsorption_energy", slice_values=self.slice_values)
        self.slice_plot_en_z = self.slice_z[2]
        self.slice_plot_z = self.slice_z[4]
        self.slice_plot_z_se = self.slice_z[9]
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

    def slice_plot_resp_surface(self, fx=None, fy=None, exp = True, density = 30, num_trial = 0, fit = True):
        #create a folder to save the plots
        self.folder_name_slice = f"{self.folder_name}/slice_images"
        if not os.path.exists(self.folder_name_slice):
            os.makedirs(self.folder_name_slice)
        self.density = density
        self.exp = exp
        self.trials_plot = self.ax_client.get_trials_data_frame()
        self.trials_plot_en = []
        for l in range(len(self.ax_client.get_model_predictions())):
            self.trials_plot_en.append(self.ax_client.get_model_predictions()[l]['adsorption_energy'][0])
        #if exp == True:    
        #    fx,fy = self.slice_values['x'], self.slice_values['y']
        #self.filename = 'knn_model.sav'
        #Load the model from a file
        #loaded_model = pickle.load(open(self.filename, 'rb'))
        
        self.jx = float(self.cell_x_max/2)
        self.jy = float(self.cell_y_max/2)
        self.jz = float(self.bulk_z_max + self.input['adsorbant_init_h'])
        x_test = np.linspace(0, self.jx, 30)
        y_test = np.linspace(0, self.jy, 30)
        z_test_test = np.linspace(self.bulk_z_max, self.jz, 30)
        #positions_test = list(itertools.product(x_test, y_test))
        #Predict energies using old model
        #E_test = loaded_model.predict(positions_test).reshape(len(x_test),len(y_test), order = 'F')
        
        #Predict energies using updated model
        fz = self.slice_values['z']
        self.fzz = [fz]
        
        #x,z fixed
        if self.input['log_data'] == 'T':
            plt.plot(self.slice_plot_y, np.log(self.slice_plot_en_y), color = "red", label = "BO model")
            plt.scatter(self.trials_plot['y'], np.log(self.trials_plot_en), color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['y'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_y, np.log(self.slice_plot_en_y) - np.log(self.slice_plot_y_se), np.log(self.slice_plot_en_y) + np.log(self.slice_plot_y_se), alpha=0.3)
        else:
            plt.plot(self.slice_plot_y, self.slice_plot_en_y, color = "red", label = "BO model")
            plt.scatter(self.trials_plot['y'], self.trials_plot_en, color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['y'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_y, self.slice_plot_en_y - self.slice_plot_y_se, self.slice_plot_en_y + self.slice_plot_y_se, alpha=0.3)
        plt.ylabel('Energy')
        plt.xlabel('y [A]')
        plt.xlim(0, np.max(self.slice_plot_y))
        plt.grid()
        plt.legend()
        plt.title(f'BO Predicted Energy Surface, x = {self.slice_values["x"]:.3f}, z = {self.slice_values["z"]:.3f}')
        if self.input['log_scale'] == 'T':
                plt.yscale('log')
        if fit == True:
            self.sl_E_test_xfix = self.sl_knn_xfix.predict(list(itertools.product(self.fxx, y_test, self.fzz))).reshape(len(y_test),len(self.fzz), order = 'F')
            if self.input['log_data'] == 'T':
                plt.plot(y_test, np.log(self.sl_E_test_xfix), color = "darkblue", label = f"Actual Energy Surface")
            else:
                plt.plot(y_test, self.sl_E_test_xfix, color = "darkblue", label = "Actual Energy Surface")
            plt.title(f'BO Predicted vs Actual Energy Surface, x = {self.slice_values["x"]:.3f}, z = {self.slice_values["z"]:.3f}')
        plt.savefig(f"{self.folder_name_slice}/xfix_slice_predfit_{num_trial}.png", dpi=300)
        if fit == True:
            plt.ylim(np.min(self.sl_E_test_xfix),np.max(self.sl_E_test_xfix))
            plt.savefig(f"{self.folder_name_slice}/xfix_slice_predfit_lim_{num_trial}.png", dpi=300)
        plt.show()

        #y,z fixed
        if self.input['log_data'] == 'T':
            plt.plot(self.slice_plot_x, np.log(self.slice_plot_en_x), color = "red", label = "BO model")
            plt.scatter(self.trials_plot['x'], np.log(self.trials_plot_en), color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['x'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_x, np.log(self.slice_plot_en_x) - np.log(self.slice_plot_x_se), np.log(self.slice_plot_en_x) + np.log(self.slice_plot_x_se), alpha=0.3)
        else:
            plt.plot(self.slice_plot_x, self.slice_plot_en_x, color = "red", label = "BO model")
            plt.scatter(self.trials_plot['x'], self.trials_plot_en, color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['x'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_x, self.slice_plot_en_x - self.slice_plot_x_se, self.slice_plot_en_x + self.slice_plot_x_se, alpha=0.3)
        plt.ylabel('Energy')
        plt.xlabel('x [A]')
        plt.xlim(0, np.max(self.slice_plot_x))
        plt.grid()
        plt.legend()
        plt.title(f'BO Predicted vs Actual Energy Surface, y = {self.slice_values["y"]:.3f}, z = {self.slice_values["z"]:.3f}')
        if self.input['log_scale'] == 'T':
                plt.yscale('log')
        if fit == True:
            self.sl_E_test_yfix = self.sl_knn_yfix.predict(list(itertools.product(x_test,self.fyy, self.fzz))).reshape(len(x_test),len(self.fzz), order = 'F')
            if self.input['log_data'] == 'T':
                plt.plot(x_test, np.log(self.sl_E_test_yfix), color = "darkblue", label = "Actual Energy Surface")
            else:
                plt.plot(x_test, self.sl_E_test_yfix, color = "darkblue", label = "Actual Energy Surface")
            plt.title(f'BO Predicted vs Actual Energy Surface, y = {self.slice_values["y"]:.3f}, z = {self.slice_values["z"]:.3f}')
        plt.savefig(f"{self.folder_name_slice}/yfix_slice_predfit_{num_trial}.png", dpi=300)
        if fit == True:
            plt.ylim(np.min(self.sl_E_test_yfix),np.max(self.sl_E_test_yfix))
            plt.savefig(f"{self.folder_name_slice}/yfix_slice_predfit_lim_{num_trial}.png", dpi=300)
        plt.show()

        #x,y fixed
        if self.input['log_data'] == 'T':
            plt.plot(self.slice_plot_z, np.log(self.slice_plot_en_z), color = "red", label = "BO model")
            plt.scatter(self.trials_plot['z'], np.log(self.trials_plot_en), color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['z'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_z, np.log(self.slice_plot_en_z) - np.log(self.slice_plot_z_se), np.log(self.slice_plot_en_z) + np.log(self.slice_plot_z_se), alpha=0.3)
        else:
            plt.plot(self.slice_plot_z, self.slice_plot_en_z, color = "red", label = "BO model")
            plt.scatter(self.trials_plot['z'], self.trials_plot_en, color = "black", label = "Trials", marker="x")
            plt.axvline(self.paratrialplot['z'], color = "green", label = "Next trial", linestyle = "--")
            plt.fill_between(self.slice_plot_z, self.slice_plot_en_z - self.slice_plot_z_se, self.slice_plot_en_z + self.slice_plot_z_se, alpha=0.3)
        plt.ylabel('Energy')
        plt.xlabel('z [A]')
        plt.xlim(np.min(self.slice_plot_z),np.max(self.slice_plot_z))
        plt.grid()
        plt.legend()
        plt.title(f'BO Predicted vs Actual Energy Surface, x = {self.slice_values["x"]:.3f}, y = {self.slice_values["y"]:.3f}')
        if self.input['log_scale'] == 'T':
                plt.yscale('log')
        if fit == True:
            self.sl_E_test_zfix = self.sl_knn_zfix.predict(list(itertools.product(self.fzz,self.fyy,z_test_test))).reshape(len(x_test),len(self.fyy), order = 'F')
            if self.input['log_data'] == 'T':
                plt.plot(z_test_test, np.log(self.sl_E_test_zfix), color = "darkblue", label = "Actual Energy Surface")
            else:
                plt.plot(z_test_test, self.sl_E_test_zfix, color = "darkblue", label = "Actual Energy Surface")
            plt.title(f'BO Predicted vs Actual Energy Surface, x = {self.slice_values["x"]:.3f}, y = {self.slice_values["y"]:.3f}')
        plt.savefig(f"{self.folder_name_slice}/zfix_slice_predfit_{num_trial}.png", dpi=300)
        if fit == True:
            plt.ylim(np.min(self.sl_E_test_zfix),np.max(self.sl_E_test_zfix))
            plt.savefig(f"{self.folder_name_slice}/zfix_slice_predfit_lim_{num_trial}.png", dpi=300)
        plt.show()

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

    def view(self):    #Easy quick view of the atoms
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

    def exp_view(self):    #Whole visualization of experiment
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
        if self.input['log_data'] == 'T':
            ax[0].plot(x_bfgs, np.log(y_bfgs), label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        else:
            ax[0].plot(x_bfgs, y_bfgs, label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        
        #BO
        trace = self.df['bo_trace']
        x = range(len(trace))
        if self.input['log_data'] == 'T':
            ax[0].plot(x, np.log(trace), label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        else:
            ax[0].plot(x, trace, label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        
        # Plot the optimization trace vs time
        ax[1].set_title('BO Optimized Adsorption vs time')
        ax[1].set_xlabel("Optimization time (s)")
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].grid(True, linestyle='--', color='0.7', zorder=-1, linewidth=1, alpha=0.5)
        #BFGS
        xt_bfgs = self.df_bfgs['Time']
        if self.input['log_data'] == 'T':
            ax[1].plot(xt_bfgs, np.log(y_bfgs), label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        else:
            ax[1].plot(xt_bfgs, y_bfgs, label=f"{self.input['calc_method']}_BFGS", color='r', marker='o', linestyle='-')
        #BO
        xt_BO = self.df['run_time']
        ax[1].axvline(x=self.df['run_time'][self.input['gs_init_steps']-1], color='k', linestyle='--', linewidth=2, alpha=0.5, label='End of initialization trials')
        if self.input['log_data'] == 'T':
            ax[1].plot(xt_BO, np.log(trace), label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        else:
            ax[1].plot(xt_BO, trace, label=f"{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}", color='b', marker='o', linestyle='-')
        plt.legend()
        ax[0].legend()
        if self.input['log_scale'] == 'T':
                ax[0].set_yscale('log')
                ax[1].set_yscale('log')
        if self.input["save_fig"] == "T":
            fig.savefig(f"{self.folder_name}/ase_ads_Opt_trace_{self.input['adsorbant_atom']}_on_{self.input['surface_atom']}_{self.input['calc_method']}_{self.input['bo_surrogate']}_{self.input['bo_acquisition_f']}_{self.curr_date_time}.png")        

    def acqf_contour_slice_gif(self):
        self.acqf_images_x = []
        for filename in os.listdir(f"{self.folder_name}/acqf_images"):
            if filename.startswith('xfix'):
                self.acqf_images_x.append(imageio.imread(f"{self.folder_name}/acqf_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/acqf_x.gif", self.acqf_images_x, fps=0.7)
        
        self.acqf_images_y = []
        for filename in os.listdir(f"{self.folder_name}/acqf_images"):
            if filename.startswith('yfix'):
                self.acqf_images_y.append(imageio.imread(f"{self.folder_name}/acqf_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/acqf_y.gif", self.acqf_images_y, fps=0.7)
        
        self.acqf_images_z = []
        for filename in os.listdir(f"{self.folder_name}/acqf_images"):
            if filename.startswith('zfix'):
                self.acqf_images_z.append(imageio.imread(f"{self.folder_name}/acqf_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/acqf_z.gif", self.acqf_images_z, fps=0.7)
        
        self.en_contour_xy_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('xy_B'):
                self.en_contour_xy_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_xy.gif", self.en_contour_xy_images, fps=0.7)
        
        self.en_contour_xy_se_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('xy_se'):
                self.en_contour_xy_se_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_xy_se.gif", self.en_contour_xy_se_images, fps=0.7)
        
        self.en_contour_xz_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('xz_B'):
                self.en_contour_xz_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_xz.gif", self.en_contour_xz_images, fps=0.7)
        
        self.en_contour_xz_se_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('xz_se'):
                self.en_contour_xz_se_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_xz_se.gif", self.en_contour_xz_se_images, fps=0.7)
        
        self.en_contour_yz_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('yz_B'):
                self.en_contour_yz_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_yz.gif", self.en_contour_yz_images, fps=0.7)
        
        self.en_contour_yz_se_images = []
        for filename in os.listdir(f"{self.folder_name}/contour_images"):
            if filename.startswith('yz_se'):
                self.en_contour_yz_se_images.append(imageio.imread(f"{self.folder_name}/contour_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/contourBO_yz_se.gif", self.en_contour_yz_se_images, fps=0.7)
        
        self.slice_images_x = []
        for filename in os.listdir(f"{self.folder_name}/slice_images"):
            if filename.startswith('xfix'):
                self.slice_images_x.append(imageio.imread(f"{self.folder_name}/slice_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/slice_x.gif", self.slice_images_x, fps=0.7)
        
        self.slice_images_y = []
        for filename in os.listdir(f"{self.folder_name}/slice_images"):
            if filename.startswith('yfix'):
                self.slice_images_y.append(imageio.imread(f"{self.folder_name}/slice_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/slice_y.gif", self.slice_images_y, fps=0.7)
        
        self.slice_images_z = []
        for filename in os.listdir(f"{self.folder_name}/slice_images"):
            if filename.startswith('zfix'):
                self.slice_images_z.append(imageio.imread(f"{self.folder_name}/slice_images/{filename}"))
        imageio.mimsave(f"{self.folder_name}/slice_z.gif", self.slice_images_z, fps=0.7)