@testsetup module TestCases
    using AtomsBase
    using DFTK
    using Unitful
    using UnitfulAtomic

    # Basic silicon system.
    a = 5.431u"angstrom"
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]];
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    system = periodic_system(lattice, atoms, positions)
end
