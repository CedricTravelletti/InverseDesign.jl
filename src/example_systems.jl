# 
# Collection of example systems used for inverse design research.
# 
using DFTK
using Unitful
using UnitfulAtomic

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
