using Test
using OpSum
using OpSum: charge
using OpSum.PauliOperators: PauliOperator
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit: Trivial, SU2Irrep, U1Irrep, sectortype

# Generic letter interface (dense→term migration, step 1): `charge` / `sectortype` must treat any
# alphabet letter uniformly — symmetric letters carry their irrep charge, dense letters are trivial.
@testset "letter interface: charge / sectortype" begin
    @testset "dense (Pauli) letters are trivial-sector" begin
        for id in 0x00:0x03
            @test charge(PauliOperator(id)) === Trivial()
        end
        @test sectortype(PauliOperator) === Trivial
    end

    @testset "symmetric (ITO) letters carry their irrep charge" begin
        @test charge(IrrepOperator(SU2Irrep(1), 1)) === SU2Irrep(1)
        @test charge(IrrepOperator(SU2Irrep(0), 1)) === SU2Irrep(0)
        @test charge(IrrepOperator(U1Irrep(-1), 1)) === U1Irrep(-1)
        @test sectortype(IrrepOperator{SU2Irrep}) === SU2Irrep
        @test sectortype(IrrepOperator{U1Irrep}) === U1Irrep
    end

    @testset "charge value inhabits the declared sector type" begin
        @test charge(PauliOperator(0x01)) isa sectortype(PauliOperator)
        @test charge(IrrepOperator(SU2Irrep(1), 1)) isa sectortype(IrrepOperator{SU2Irrep})
    end
end
