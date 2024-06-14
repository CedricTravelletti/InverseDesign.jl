# 
# Collection of example systems used for inverse design research.
# 
using DFTK
setup_threading()
using Unitful
using UnitfulAtomic
using ForwardDiff
using ComponentArrays


function construct_silicon()
    a = 5.431u"angstrom"
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]];
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    periodic_system(lattice, atoms, positions)
end

function construct_diamond()
    a = 3.567u"angstrom"
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]];
    C = ElementPsp(:C; psp=load_psp("hgh/lda/c-q4"))
    atoms     = [C, C]
    positions = [ones(3)/8, -ones(3)/8]
    periodic_system(lattice, atoms, positions)
end

function construct_gammaP_bandgap_vs_strain12()
    setup_threading()
    system = construct_silicon()

    model_kwargs = (; functionals = [:lda_x, :lda_c_pw], temperature = 1e-4)
    basis_kwargs = (; kgrid = [4, 4, 4], Ecut = 30.0)
    scf_kwargs = (; tol = 1e-7) # Tight convergence required for forwarddiff
    calculator = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs, verbose=true)
	
    x0 = Vector(flatten(DFTK.parse_system(system).positions))
	
    """ Helper function for gamma point bandgap versus the first two components 
    of the strain. Everything done around equilibrium positions. 
    
    """
    function _f_strain12(strain12)
    	strain = [strain12; [0., 0, 0, 0]]
    	positions_flat = ComponentVector(; atoms=x0, strain)
	gamma_point_bandgap(calculator, system, positions_flat)[:bandgap]
    end
    
    function gammaP_bandgap_vs_strain12(strain12)
    	y = _f_strain12(strain12)
    	y_grad = ForwardDiff.gradient(_f_strain12, strain12)
    	[[y]; y_grad]
    end
end
