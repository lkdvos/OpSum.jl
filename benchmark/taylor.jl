# Build time of the Taylor-cluster time-evolution MPO on infinite chains: OpSum's symbolic
# `taylor_mpo(H, N)` + `instantiate(·, τ)` against MPSKit's `make_time_mpo(H, dt, TaylorCluster(; N))`
# (which needs `instantiate(mpo(H))` first; that is timed separately and not charged to either side).
#
#   julia --project=benchmark benchmark/taylor.jl
#
# Environment: OPSUM_BENCH_QUICK=1 for a reduced sweep; OPSUM_BENCH_LIMIT (seconds, default 120) after which
# MPSKit is skipped when it is predicted to exceed it.

using OpSum
using OpSum: mpo, taylor_mpo, instantiate, opsum
using MPSKit
using TensorKit
using Printf

const QUICK = get(ENV, "OPSUM_BENCH_QUICK", "0") == "1"
const LIMIT = parse(Float64, get(ENV, "OPSUM_BENCH_LIMIT", "120"))
const DT = 0.1

# Ising on a cylinder of circumference Ly: `-Σ Zᵢ Zᵢ₊₁` around a ring (open, as the unit cell), `-Σ Zᵢ Zᵢ₊Ly`
# between rings, and a transverse field.
function ising_cylinder(Ly; J = 1.0, h = 0.7)
    S = spin_ops(Trivial)
    return opsum(
        InfiniteChain(fill(spin_basis(Trivial), Ly)),
        (-J * S.S_z_S_z[i, i + 1] for i in 1:(Ly - 1)),
        (-J * S.S_z_S_z[i, i + Ly] for i in 1:Ly),
        (h * S.S_x[i] for i in 1:Ly)
    )
end

# U(1) XXZ chain with couplings out to range `k`, `J_r = 1/r`.
function xxz_long(k; Δ = 0.8)
    U = spin_ops(U1Irrep)
    return opsum(
        InfiniteChain(spin_basis(U1Irrep)),
        ((0.5 / r) * (U.S_plus_S_min[1, 1 + r] + U.S_min_S_plus[1, 1 + r]) + (Δ / r) * U.S_z_S_z[1, 1 + r] for r in 1:k)
    )
end

# smallest of `k` runs, after one warm-up; a single run once it is slow
function besttime(f; k = 3)
    t = @elapsed f()
    t > 5 && return t
    for _ in 2:k
        t = min(t, @elapsed f())
    end
    return t
end

bonddim(O) = sum(i -> dim(MPSKit.right_virtualspace(O, i)), 1:length(O)) ÷ length(O)

# `last[N]` is the MPSKit time at `N` on the previous (smaller) model; the next one is predicted to be at least
# 10x slower, and MPSKit is skipped when that exceeds `LIMIT`. Likewise from `N - 1` to `N` on one model (25x).
function run_model(name, H, Ns, last)
    Hm = instantiate(mpo(H))
    prev = 0.0
    for N in Ns
        skip = max(10 * get(last, N, 0.0), 25 * prev) > LIMIT
        τ = -DT
        ours() = instantiate(taylor_mpo(H, N), τ)
        ours()  # compile
        t1 = besttime(ours)
        D1 = bonddim(ours())
        if skip
            @printf("%-14s N=%d  OpSum %9.4f s (D=%d)   MPSKit skipped\n", name, N, t1, D1)
            flush(stdout)
            continue
        end
        ref() = make_time_mpo(Hm, DT, TaylorCluster(; N); imaginary_evolution = true)
        t2 = @elapsed (O = ref())
        D2 = bonddim(O)
        t2 < 5 && (t2 = min(t2, besttime(ref)))
        last[N] = prev = t2
        @printf("%-14s N=%d  OpSum %9.4f s (D=%d)   MPSKit %9.4f s (D=%d)   speedup %.1fx\n", name, N, t1, D1, t2, D2, t2 / t1)
        flush(stdout)
    end
    return nothing
end

# compile both sides on a small model, so no timing below includes it
for H in (ising_cylinder(2), xxz_long(1)), N in 1:2
    instantiate(taylor_mpo(H, N), -DT)
    make_time_mpo(instantiate(mpo(H)), DT, TaylorCluster(; N); imaginary_evolution = true)
end

let Ns = 1:4
    skip = Dict{Int, Float64}()
    for Ly in (QUICK ? (2, 3) : 2:6)
        run_model("ising Ly=$Ly", ising_cylinder(Ly), Ns, skip)
    end
    skip = Dict{Int, Float64}()
    for k in (QUICK ? (1, 2) : 1:4)
        run_model("xxz-U1 k=$k", xxz_long(k), Ns, skip)
    end
end
