module OneDimensionalTDHF

using Plots
using LinearAlgebra
using Parameters
using OneDimensionalStaticHF


@with_kw struct PhysicalParam{T} @deftype Float64
    mc² = 938.
    ħc  = 197. 
    
    t₀ = -497.726 
    t₃ = 17_270.
    
    a  = 0.45979 
    V₀ = -166.9239/a 
    
    ρ₀ = 0.16
    σ = 1.4

    Δz = 0.1
    Nz::Int64 = 200; 

    zs::T = range((-Nz/2+1/2)*Δz, (Nz/2-1/2)*Δz, length=Nz)
    
    cnvl_coeff::Vector{Float64} = calc_cnvl_coeff(Δz, a)
    
    ψs₀::Matrix{ComplexF64}
    spEs₀::Vector{Float64}
    Πs₀::Vector{Float64}
    Efermi₀ 
end


σ = 2.0 
Δz = 0.1
Nz = 600
ψs₀, spEs₀, Πs₀, Efermi₀, ρ₀, τ₀ = HF_calc_with_imaginary_time_step(
    σ=σ, Δz=Δz, Nz=div(Nz,2), show=false)

param = PhysicalParam(
    σ=σ, Δz=Δz, Nz=Nz,
    ψs₀=ψs₀, spEs₀=spEs₀, Πs₀=Πs₀, Efermi₀=Efermi₀)

#@show param.zs param.zs[div(Nz,2)+1]



function initial_states(param, z₀, S; Nslab=1)
    @assert length(z₀) === Nslab
    @unpack mc², ħc, Nz, Δz, zs, ψs₀, spEs₀, Efermi₀ = param
    
    nstates₀ = size(ψs₀, 2)
    ψs = zeros(ComplexF64, Nz, nstates₀*Nslab)
    occ = zeros(Float64, nstates₀*Nslab)
    ψ = zeros(ComplexF64, Nz)
    
    for islab in 1:Nslab
        dz₀ = floor(Int, z₀[islab]/Δz)
        
        for istate₀ in 1:nstates₀
            i = (islab-1)*nstates₀ + istate₀
            
            ψs[1:div(Nz,2),i] = reverse(ψs₀[:,istate₀])*Πs₀[istate₀]
            ψs[1+div(Nz,2):Nz,i] = ψs₀[:,istate₀]
            
            for iz in 1:Nz
                jz = iz - dz₀
                if jz < 1 || jz > Nz
                    ψ[iz] = 0
                    continue
                end
                ψ[iz] = ψs[jz,i]
            end
            
            @. ψs[:,i] = ψ*exp(im*S)
            if spEs₀[istate₀] ≤ Efermi₀
                occ[i] = (2mc²/(π*ħc*ħc))*(Efermi₀-spEs₀[istate₀])
            end
        end
    end
    
    return ψs, occ
end


function test_initial_states(param, z₀; Nslab=1)
    @unpack Nz, zs = param
    
    S = zeros(Float64, Nz)
    #S = @. zs^2
    @time ψs, occ = initial_states(param, z₀, S; Nslab=Nslab)
    
    
    p = plot()
    for i in 1:size(ψs, 2)
        plot!(p, zs, @views @. real(ψs[:,i]))
    end
    display(p)
    
    @show sum(occ)/2
    occ
end

#test_initial_states(param, [-10, 10]; Nslab=2)

function make_Hamiltonian(param, vpot)
    @unpack Δz, Nz, zs = param
    
    dv = similar(zs)
    @. dv = 2/Δz^2 + vpot
    
    ev = fill(-1/Δz^2, Nz-1)
    
    return SymTridiagonal(dv, ev)
end

function test_make_Hamiltonian(param; σ=1.4)
    @unpack zs = param

    vpot = @. zs^2
    Hmat = make_Hamiltonian(param, vpot)
    
    vals, vecs = eigen(Hmat)
    vals[1:10] ./ 2
end

#test_make_Hamiltonian(param)

function first_deriv!(dψ, zs, ψ)
    Nz = length(zs)
    Δz = zs[2] - zs[1]
    
    dψ[1] = ψ[2]/2Δz
    for iz in 2:Nz-1
        dψ[iz] = (ψ[iz+1] - ψ[iz-1])/2Δz
    end
    dψ[Nz] = -ψ[Nz-1]/2Δz
    
    return
end

function test_first_deriv!(param)
    @unpack zs = param

    ψ = @. exp(-0.5zs*zs)
    
    dψ = similar(zs)
    first_deriv!(dψ, zs, ψ)
    
    dψ_exact = @. -zs*exp(-0.5zs*zs)
    
    plot(zs, dψ)
    plot!(zs, dψ_exact)
end

#test_first_deriv!(param)

function calc_density!(ρ, τ, param, ψs, occ)
    @unpack mc², ħc, Δz, Nz, zs = param
    nstates = size(ψs, 2)
    
    fill!(ρ, 0)
    fill!(τ, 0)
    dψ = zeros(ComplexF64, Nz)
    for i in 1:nstates
        @views ψ = ψs[:,i]
        first_deriv!(dψ, zs, ψ)
        @. ρ += occ[i]*real(dot(ψ, ψ))
        @. τ += occ[i]*real(dot(dψ, dψ))
        @. τ += (π/2)*occ[i]^2*real(dot(ψ, ψ))
    end
end


function test_calc_density!(param)
    @unpack Nz, Δz, zs = param
    
    S = zeros(Float64, Nz)
    @time ψs, occ = initial_states(param, [-10,10], S; Nslab=2)
    
    ρ = similar(zs)
    τ = similar(zs)
    calc_density!(ρ, τ, param, ψs, occ)
    
    @show sum(ρ)*Δz/2
    
    p = plot(ylim=(0,0.3))
    plot!(zs, ρ)
    plot!(zs, τ)
    display(p)
    
    vpot = similar(zs)
    calc_potential!(vpot, param, ρ)
    p = plot(zs, vpot)
    display(p)
end

#test_calc_density!(param)

function calc_norm(zs, ψ)
    Δz = zs[2] - zs[1]
    sqrt(dot(ψ, ψ)*Δz)
end

function calc_sp_energy(param, Hmat, ψ)
    @unpack ħc, mc² = param
    return dot(ψ, Hmat, ψ)/dot(ψ, ψ) * (ħc^2/2mc²)
end

function real_time_evolution!(ψs, ψs_mid, occ, 
        ρ, τ, ρ_mid, τ_mid, vpot, vpot_mid, param; Δt=0.1)
    
    @unpack Nz, Δz, zs = param
    nstates = size(ψs, 2)
    
    calc_potential!(vpot, param, ρ)
    Hmat = make_Hamiltonian(param, vpot)
    
    U₁ =          (I - 0.5*im*Δt*Hmat)
    U₂ = factorize(I + 0.5*im*Δt*Hmat)
    
    for i in 1:nstates
        @views ψs_mid[:,i] = U₂\(U₁*ψs[:,i])
    end
    
    calc_density!(ρ_mid, τ_mid, param, ψs_mid, occ)
    calc_potential!(vpot_mid, param, ρ)
    Hmat_mid = make_Hamiltonian(param, vpot_mid)
    @. Hmat = (Hmat + Hmat_mid)/2
    
    U₁ =          (I - 0.5*im*Δt*Hmat)
    U₂ = factorize(I + 0.5*im*Δt*Hmat)
    
    for i in 1:nstates
        @views ψs[:,i] = U₂\(U₁*ψs[:,i])
    end
    
    #@views ψs[:,i] ./= calc_norm(zs, ψs[:,i])
    
    #@show calc_norm(zs, ψs[:,1]) 
end

function test_real_time_evolution!(param; α=0.1, Δt=0.1, T=20)
    @unpack Nz, zs = param
    
    S = zeros(Float64, Nz)
    #@. S = α*zs^2
    @time ψs, occ = initial_states(param, 0.0, S; Nslab=1)
    
    ρ = similar(zs)
    τ = similar(zs)
    vpot = similar(zs)
    calc_density!(ρ, τ, param, ψs, occ)
    
    ψs_mid = similar(ψs)
    ρ_mid = similar(zs)
    τ_mid = similar(zs)
    vpot_mid = similar(zs)
    
    anim = @animate for it in 1:floor(Int, T/abs(Δt))
        real_time_evolution!(ψs, ψs_mid, occ, 
            ρ, τ, ρ_mid, τ_mid, vpot, vpot_mid, param; Δt=Δt)
        calc_density!(ρ, τ, param, ψs, occ)
        plot(zs, ρ; ylim=(0,0.3))
    end
    
    gif(anim, "anim_fps15.gif", fps = 15)
end


function small_amplitude_dynamics(;σ=1.4, Δz=0.1, Nz=600, α=0.02, Δt=0.025, T=20)

    ψs₀, spEs₀, Πs₀, Efermi₀, ρ₀, τ₀ = HF_calc_with_imaginary_time_step(
        σ=σ, Δz=Δz, Nz=div(Nz,2), show=false)

    param = PhysicalParam(
        σ=σ, Δz=Δz, Nz=Nz,
        ψs₀=ψs₀, spEs₀=spEs₀, Πs₀=Πs₀, Efermi₀=Efermi₀)

    @unpack Nz, zs = param
    S = zeros(Float64, Nz)
    @. S = α*zs^2
    @time ψs, occ = initial_states(param, 0.0, S; Nslab=1)
    
    ρ = similar(zs)
    τ = similar(zs)
    vpot = similar(zs)
    calc_density!(ρ, τ, param, ψs, occ)
    
    ψs_mid = similar(ψs)
    ρ_mid = similar(zs)
    τ_mid = similar(zs)
    vpot_mid = similar(zs)
    
    anim = @animate for it in 1:floor(Int, T/abs(Δt))
        real_time_evolution!(ψs, ψs_mid, occ, 
            ρ, τ, ρ_mid, τ_mid, vpot, vpot_mid, param; Δt=Δt)
        calc_density!(ρ, τ, param, ψs, occ)
        plot(zs, ρ; ylim=(0,0.3))
    end
    
    gif(anim, "anim_fps15.gif", fps = 15)
end


function slab_collision(;σ=1.4, Δz=0.1, Nz=600, α=0.02, Δt=0.025, T=20)
    
end







#test_real_time_evolution!(param; α=0.0, Δt=0.01, T=10)

end # module
