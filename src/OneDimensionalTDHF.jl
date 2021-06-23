module OneDimensionalTDHF

using Plots
using LinearAlgebra
using Parameters
using OneDimensionalStaticHF
using Statistics
using SparseArrays
using Arpack
using LsqFit


@with_kw struct PhysicalParam{T} @deftype Float64
    mc² = 938.
    ħc  = 197. 
    
    t₀ = -497.726 
    t₃ = 17_270.
    
    a  = 0.45979 
    V₀ = -166.9239/a 

    V_ext = 0
    a_ext = 2
    
    ρ₀ = 0.16

    Nslab::Int64 = 2
    σ::Vector{Float64} = fill(1.4, Nslab); @assert length(σ) === Nslab

    Δz = 0.1
    Nz::Int64 = 200; @assert iseven(Nz)

    zs::T = range((-Nz/2+1/2)*Δz, (Nz/2-1/2)*Δz, length=Nz)
    
    cnvl_coeff::Vector{Float64} = calc_cnvl_coeff(Δz, a)
end

@with_kw struct Densities 
    "ρ: number density"
    ρ::Vector{Float64}

    "τ: kinetic density"
    τ::Vector{Float64} = similar(ρ)
end

@with_kw struct SingleParticleStates 
    "nstates: number of single particle states"
    nstates::Int64

    "ψs: wave functions"
    ψs::Matrix{ComplexF64}; @assert size(ψs, 2) === nstates 

    "occ: occupation numbers"
    occ::Vector{Float64}; @assert length(occ) === nstates
end




function calc_potential!(vpot, param, dens, Lmat)
    OneDimensionalStaticHF.calc_potential!(vpot, param, dens, Lmat)

    @unpack mc², ħc, V_ext, a_ext, zs = param
    @. vpot += V_ext*exp(-zs^2/2a_ext^2)*2mc²/ħc^2
    
    return 
end




function initial_states(param, z₀, S)
    @unpack mc², ħc, Nslab, σ, Nz, Δz, zs = param
    @assert length(z₀) === Nslab === size(S, 2)

    states_static = OneDimensionalStaticHF.SingleParticleStates[]
    nstates_static = zeros(Int64, Nslab)

    for islab in 1:Nslab 
        states, dens = HF_calc_with_imaginary_time_step(;σ=σ[islab], Nz=div(Nz,2))
        push!(states_static, states)
        
        nstates_static[islab] = states_static[islab].nstates 
    end

    nstates = sum(nstates_static)

    ψ₀  = zeros(ComplexF64, div(Nz,2)) # storing static wave functions 
    ψ   = zeros(ComplexF64, Nz) # work array for translating wave functions 

    ψs  = zeros(ComplexF64, Nz, nstates)
    occ = zeros(Float64, nstates)

    istate = 0
    for islab in 1:Nslab
        dz₀ = floor(Int, z₀[islab]/Δz)
        
        for istate_static in 1:nstates_static[islab]
            istate += 1

            ψ₀[:] = states_static[islab].ψs[:,istate_static]
            Π₀    = states_static[islab].Πs[istate_static]
            
            ψs[1:div(Nz,2),istate] = reverse(ψ₀)*Π₀
            ψs[1+div(Nz,2):Nz,istate] = ψ₀
            
            for iz in 1:Nz
                jz = iz - dz₀
                if jz < 1 || jz > Nz
                    ψ[iz] = 0
                    continue
                end
                ψ[iz] = ψs[jz,istate]
            end
            
            @. ψs[:,istate] = ψ*exp(im*S[:,islab])
            occ[istate] = states_static[islab].occ[istate_static]
        end
    end

    states = SingleParticleStates(nstates=nstates, ψs=ψs, occ=occ)

    return states 
end



function test_initial_states(param, z₀)
    @unpack Nz, zs, Nslab = param
    
    S = zeros(Float64, Nz, Nslab)
    @time states = initial_states(param, z₀, S)
    
    @unpack nstates, ψs = states 
    
    p = plot(legend=false)
    for i in 1:nstates 
        plot!(p, zs, @views @. real(ψs[:,i]))
    end
    display(p)

end


#=
function make_Hamiltonian!(Hmat, param, vpot)
    @unpack Δz, Nz, zs = param
    @unpack dv, ev = Hmat 
    
    @. dv = 2/Δz^2 + vpot
    
    @. ev = -1/Δz^2
    
    return 
end

function test_make_Hamiltonian!(param; σ=1.4)
    @unpack zs, Nz = param

    vpot = @. zs^2

    dv = zeros(Float64, Nz)
    ev = zeros(Float64, Nz-1)
    Hmat = SymTridiagonal(dv, ev)
    make_Hamiltonian!(Hmat, param, vpot)
    
    vals, vecs = eigen(Hmat)
    vals[1:10] ./ 2
end
=#


@inline function second_deriv_coeff(i, j, a, N) 
    d = 0.0 
    if i === 1
        d += ifelse(j===3, -1/12, 0)
        d += ifelse(j===2,   4/3, 0)
        d += ifelse(j===1,  -5/2, 0)
    elseif i === 2
        d += ifelse(j===4, -1/12, 0)
        d += ifelse(j===3, 4/3, 0)
        d += ifelse(j===2, -5/2, 0)
        d += ifelse(j===1, 4/3, 0)
    elseif i === N-1
        d += ifelse(j===N, 4/3, 0)
        d += ifelse(j===N-1, -5/2, 0)
        d += ifelse(j===N-2, 4/3, 0)
        d += ifelse(j===N-3, -1/12, 0)
    elseif i === N 
        d += ifelse(j===N, -5/2, 0)
        d += ifelse(j===N-1, 4/3, 0)
        d += ifelse(j===N-2, -1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1, 4/3, 0)
        d += ifelse(j===i, -5/2, 0)
        d += ifelse(j===i-1, 4/3, 0)
        d += ifelse(j===i-2, -1/12, 0)
    end
    d /= a*a 
    return d 
end

function make_Hamiltonian!(Hmat, param, vpot)
    @unpack Nz, Δz, zs = param

    fill!(Hmat, 0)
    for iz in 1:Nz
        Hmat[iz,iz] += vpot[iz]

        for dz in -2:2
            jz = iz + dz 

            if !(1 ≤ jz ≤ Nz) continue end 
            Hmat[iz,jz] += -second_deriv_coeff(iz, jz, Δz, Nz)
        end
    end

    return Hmat
end

function test_make_Hamiltonian!(param; σ=1.4)
    @unpack zs, Nz = param

    vpot = @. zs^2

    Hmat = spzeros(Float64, Nz, Nz)
    make_Hamiltonian!(Hmat, param, vpot)
    
    vals, vecs = eigs(Hmat, which=:SM, nev=10)
    vals ./ 2
end

function make_Laplacian!(Lmat, param, a)
    @unpack Nz, Δz, zs = param

    fill!(Lmat, 0)
    for iz in 1:Nz
        Lmat[iz,iz] += 1/(a*a)
        for dz in -2:2
            jz = iz + dz 

            if !(1 ≤ jz ≤ Nz) continue end 
            Lmat[iz,jz] += -second_deriv_coeff(iz, jz, Δz, Nz)
        end
    end

    return Lmat
end

function test_make_Laplacian!(; a=100)
    param = PhysicalParam()
    @unpack zs, Nz = param

    Lmat = spzeros(Float64, Nz, Nz)
    @time make_Laplacian!(Lmat, param, a)

    ρ = zeros(Float64, Nz)
    @. ρ = (1 - zs^2)*exp(-zs^2/2)
    
    ϕ_exact = zeros(Float64, Nz)
    @. ϕ_exact = exp(-zs^2/2)
    #@. ϕ_exact -= ϕ_exact[end]

    ϕ = zeros(Float64, Nz)
    ϕ = Lmat\ρ

    plot(zs, ϕ)
    plot!(zs, ϕ_exact)
end


#=
function first_deriv!(dψ, param, ψ)
    @unpack Nz, Δz, zs = param 
    Nz = length(zs)
    
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
    first_deriv!(dψ, param, ψ)
    
    dψ_exact = @. -zs*exp(-0.5zs*zs)
    
    plot(zs, dψ)
    plot!(zs, dψ_exact)
end
=#


@inline function first_deriv_coeff(i, j, a, N) 
    d = 0.0 
    if i === 1
        d += ifelse(j===3, -1/12, 0)
        d += ifelse(j===2,   2/3, 0)
    elseif i === 2
        d += ifelse(j===4, -1/12, 0)
        d += ifelse(j===3,   2/3, 0)
        d += ifelse(j===1,  -2/3, 0)
    elseif i === N-1
        d += ifelse(j===N,    2/3, 0)
        d += ifelse(j===N-2, -2/3, 0)
        d += ifelse(j===N-3, 1/12, 0)
    elseif i === N 
        d += ifelse(j===N-1, -2/3, 0)
        d += ifelse(j===N-2, 1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1,   2/3, 0)
        d += ifelse(j===i-1,  -2/3, 0)
        d += ifelse(j===i-2,  1/12, 0)
    end
    d /= a
    return d 
end


function first_deriv!(dψ, param, ψ)
    @unpack Nz, Δz, zs = param 
    Nz = length(zs)
    
    fill!(dψ, 0)
    for iz in 1:Nz
        for dz in -2:2 
            jz = iz + dz
            if !(1 ≤ jz ≤ Nz) continue end 

            dψ[iz] += first_deriv_coeff(iz, jz, Δz, Nz)*ψ[jz]
        end
    end
    
    return
end

function test_first_deriv!(param)
    @unpack zs = param

    ψ = @. exp(-0.5zs*zs)
    
    dψ = similar(zs)
    first_deriv!(dψ, param, ψ)
    
    dψ_exact = @. -zs*exp(-0.5zs*zs)
    
    plot(zs, dψ)
    plot!(zs, dψ_exact)
end



function calc_density!(dψ, dens, param, states)
    @unpack mc², ħc, Δz, Nz, zs = param
    @unpack ρ, τ = dens 
    @unpack nstates, ψs, occ = states 
    
    fill!(ρ, 0)
    fill!(τ, 0)
    for i in 1:nstates
        @views ψ = ψs[:,i]
        first_deriv!(dψ, param, ψ)
        @. ρ += occ[i]*real(dot(ψ, ψ))
        @. τ += occ[i]*real(dot(dψ, dψ))
        @. τ += (π/2)*occ[i]^2*real(dot(ψ, ψ))
    end
end


function test_calc_density!(param, z₀)
    @unpack Nz, Δz, zs = param
    
    S = zeros(Float64, Nz, 2)
    @time states = initial_states(param, z₀, S)
    
    dens = Densities(ρ=similar(zs))
    dψ = zeros(ComplexF64, Nz)
    calc_density!(dψ, dens, param, states)
    
    @show sum(dens.ρ)*Δz/2
    
    p = plot(ylim=(0,0.3))
    plot!(zs, dens.ρ)
    plot!(zs, dens.τ)
    display(p)

    Lmat = spzeros(Float64, Nz, Nz)
    @time make_Laplacian!(Lmat, param, param.a)
    
    vpot = similar(zs)
    calc_potential!(vpot, param, dens, Lmat)
    p = plot(zs, vpot)
    display(p)
end





function calc_total_energy(param, dens, Lmat)
    E = OneDimensionalStaticHF.calc_total_energy(param, dens, Lmat) / 2

    @unpack mc², ħc, zs, Δz, Nz, V_ext, a_ext = param
    @unpack ρ = dens 
    ε = zeros(Float64, Nz)
    @. ε = V_ext*exp(-zs^2/2a_ext^2)*ρ

    E += sum(ε)*Δz
end

#=
function real_time_evolution!(states, states_mid, 
        dψ, dens, dens_mid, vpot, vpot_mid, Hmat, Hmat_mid, param, Lmat; Δt=0.1
    )

    @unpack mc², ħc, Nz, Δz, zs = param
    @unpack nstates, ψs = states 

    calc_potential!(vpot, param, dens, Lmat)
    make_Hamiltonian!(Hmat, param, vpot)

    U₁ =          (I - 0.5*im*(Δt/2)*ħc^2/2mc²*Hmat)
    U₂ = factorize(I + 0.5*im*(Δt/2)*ħc^2/2mc²*Hmat)

    for i in 1:nstates
        @views states_mid.ψs[:,i] = U₂\(U₁*ψs[:,i])
    end

    calc_density!(dψ, dens_mid, param, states_mid)
    calc_potential!(vpot_mid, param, dens_mid, Lmat)
    make_Hamiltonian!(Hmat, param, vpot_mid)

    # @. Hmat.dv = (Hmat.dv + Hmat_mid.dv)/2
    # @. Hmat.ev = (Hmat.ev + Hmat_mid.ev)/2

    U₁ =          (I - 0.5*im*Δt*ħc^2/2mc²*Hmat)
    U₂ = factorize(I + 0.5*im*Δt*ħc^2/2mc²*Hmat)

    for i in 1:nstates
        @views ψs[:,i] = U₂\(U₁*ψs[:,i])
    end

    calc_density!(dψ, dens, param, states)

    return
end
=#


function real_time_evolution!(states, states_mid, 
        dψ, dens, dens_mid, vpot, vpot_mid, Hmat, Hmat_mid, param, Lmat; Δt=0.1
    )

    @unpack mc², ħc, Nz, Δz, zs = param
    @unpack nstates, ψs = states 

    calc_potential!(vpot, param, dens, Lmat)
    make_Hamiltonian!(Hmat, param, vpot)

    U₁ =          (I - 0.5*im*(Δt/2)*ħc^2/2mc²*Hmat)
    U₂ = factorize(I + 0.5*im*(Δt/2)*ħc^2/2mc²*Hmat)

    for i in 1:nstates
        @views states_mid.ψs[:,i] = U₂\(U₁*ψs[:,i])
    end

    calc_density!(dψ, dens_mid, param, states_mid)
    calc_potential!(vpot_mid, param, dens_mid, Lmat)
    make_Hamiltonian!(Hmat, param, vpot_mid)

    #@. Hmat = (Hmat + Hmat_mid)/2

    U₁ =          (I - 0.5*im*Δt*ħc^2/2mc²*Hmat)
    U₂ = factorize(I + 0.5*im*Δt*ħc^2/2mc²*Hmat)

    for i in 1:nstates
        @views ψs[:,i] = U₂\(U₁*ψs[:,i])
    end

    calc_density!(dψ, dens, param, states)

    return 
end



function calc_root_mean_square_length(param, dens)
    @unpack zs = param 
    @unpack ρ = dens 

    L = sqrt(sum(@. zs^2*ρ)/sum(@. ρ))
end







function small_amplitude_dynamics(;
        σ=1.4, Δz=0.1, Nz=400, α=0.02, Δt=1e-3, T=2, save_anim=false
    )

    param = PhysicalParam(Nslab=1, σ=[σ], Δz=Δz, Nz=Nz)
    @unpack mc², ħc, zs, Nz, Δz = param

    Lmat = spzeros(Float64, Nz, Nz)
    @time make_Laplacian!(Lmat, param, param.a)

    ts = Δt:Δt:T # time [MeV⁻¹]

    dψ = zeros(ComplexF64, Nz) # first derivative of wave functions 
    Etots = zeros(Float64, length(ts)) # total energies at each time 
    Ls = zeros(Float64, length(ts)) # mean square lengths at each time 

    S = zeros(Float64, Nz)
    @. S = α*zs^2 

    states = initial_states(param, 0, S)
    dens = Densities(ρ=similar(zs))
    vpot = similar(zs)
    calc_density!(dψ, dens, param, states)

    Hmat = spzeros(Float64, Nz, Nz)
    
    states_mid = initial_states(param, 0, S)
    dens_mid = Densities(ρ=similar(zs))
    vpot_mid = similar(zs)

    Hmat_mid = spzeros(Float64, Nz, Nz)

    
    anim = @animate for it in 1:length(ts)
        real_time_evolution!(states, states_mid, 
            dψ, dens, dens_mid, vpot, vpot_mid, Hmat, Hmat_mid, param, Lmat; Δt=Δt
        )
        Etots[it] = calc_total_energy(param, dens, Lmat)
        Ls[it] = calc_root_mean_square_length(param, dens)
        if save_anim
            plot(zs, dens.ρ; ylim=(0,0.3), xlabel="z [fm]", ylabel="ρ [fm⁻³]", legend=false)
        end
    end

    
    function model(t, p)
        A, ħω, α, L₀ = p 
        @. A*cos(ħω*t + α) + L₀
    end

    p0 = Float64[0.26, 15.5, -1.7, 3.1]
    fit = curve_fit(model, ts, Ls, p0)

    p = plot(ts, Etots; xlabel="time [MeV⁻¹]", ylabel="total energy [MeV]", label=false)
    display(p)

    p = plot(ts, Ls; xlabel="time [MeV⁻¹]", ylabel="root mean square length [fm]", label=false)
    plot!(ts, model(ts, fit.param); label="model")
    display(p)

    println("")
    @show E_fluc = abs(std(Etots)/mean(Etots))*100
    @show L_fluc = abs(std(Ls)/mean(Ls))*100
    @show fit.param[2]


    if save_anim
        gif(anim, "./1dimTDHF_figure/small_amplitude_dynamics.gif", fps = 15)
    end 
end







function slab_propagation(;
        σ=1.4, z₀=0.0, Δz=0.1, Nz=600, Ecm=10, V_ext=10, a_ext=2, 
        Δt=0.01, T=1, save_anim=false
    )

    param = PhysicalParam(Nslab=1, σ=[σ], Δz=Δz, Nz=Nz, V_ext=V_ext, a_ext=a_ext)
    @unpack mc², ħc, zs, Nz, Δz = param

    Lmat = spzeros(Float64, Nz, Nz)
    @time make_Laplacian!(Lmat, param, param.a)

    ts = Δt:Δt:T # time [MeV⁻¹]

    vpot_ext = @. 0.1*(V_ext/10)*exp(-zs^2/2a_ext^2)

    dψ = zeros(ComplexF64, Nz) # first derivative of wave functions 
    Etots = zeros(Float64, length(ts)) # total energies at each time 

    k = sqrt(2mc²*Ecm/ħc^2)
    S = zeros(Float64, Nz)
    @. S = k*zs 

    states = initial_states(param, z₀, S)
    dens = Densities(ρ=similar(zs))
    vpot = similar(zs)
    calc_density!(dψ, dens, param, states)

    Hmat = spzeros(Float64, Nz, Nz)

    states_mid = initial_states(param, z₀, S)
    dens_mid = Densities(ρ=similar(zs))
    vpot_mid = similar(zs)

    Hmat_mid = spzeros(Float64, Nz, Nz)


    anim = @animate for it in 1:length(ts)
        real_time_evolution!(states, states_mid, 
            dψ, dens, dens_mid, vpot, vpot_mid, Hmat, Hmat_mid, param, Lmat; Δt=Δt
        )
        Etots[it] = calc_total_energy(param, dens, Lmat)
        if save_anim
            plot(zs, dens.ρ; ylim=(0,0.3), xlabel="z [fm]", ylabel="ρ [fm⁻³]", label="ρ")
            plot!(zs, vpot_ext; label="Vext")
        end
    end

    p = plot(ts, Etots; xlabel="time [MeV⁻¹]", ylabel="total energy [MeV]", label=false)
    display(p)

    println("")
    @show E_fluc = abs(std(Etots)/mean(Etots))*100

    if save_anim
        gif(anim, "./1dimTDHF_figure/slab_propagation.gif", fps = 15)
    end 
end






function calc_separation_distance(param, dens)
    @unpack zs = param 
    @unpack ρ = dens 

    d = 2*sqrt(sum(@. abs(zs)*ρ)/sum(@. ρ))
end


function slab_collision(;
        σ=1.4, z₀=[-15.0, 15.0], Ecm=10,
        Δz=0.1, Nz=600, Δt=0.005, T=1, save_anim=false
    )

    param = PhysicalParam(Nslab=2, σ=[σ, σ], Δz=Δz, Nz=Nz)
    @unpack mc², ħc, zs, Nz, Δz = param

    Lmat = spzeros(Float64, Nz, Nz)
    @time make_Laplacian!(Lmat, param, param.a)

    ts = Δt:Δt:T # time [MeV⁻¹]

    dψ = zeros(ComplexF64, Nz) # first derivative of wave functions 
    Etots = zeros(Float64, length(ts)) # total energies at each time 

    Ls = zeros(Float64, length(ts)) # mean square lengths at each time 
    ds = zeros(Float64, length(ts)) # separation distance at each time 
    σs = zeros(Float64, length(ts)) # fragment elongation at each time 

    k = sqrt(2mc²*Ecm/ħc^2)
    S = zeros(Float64, Nz, 2)
    @. S[:,1] =  k*zs 
    @. S[:,2] = -k*zs

    states = initial_states(param, z₀, S)
    dens = Densities(ρ=similar(zs))
    vpot = similar(zs)
    calc_density!(dψ, dens, param, states)

    Hmat = spzeros(Float64, Nz, Nz)
    
    states_mid = initial_states(param, z₀, S)
    dens_mid = Densities(ρ=similar(zs))
    vpot_mid = similar(zs)

    Hmat_mid = spzeros(Float64, Nz, Nz)

    
    anim = @animate for it in 1:length(ts)
        real_time_evolution!(states, states_mid, 
            dψ, dens, dens_mid, vpot, vpot_mid, Hmat, Hmat_mid, param, Lmat; Δt=Δt
        )
        Etots[it] = calc_total_energy(param, dens, Lmat)
        Ls[it] = calc_root_mean_square_length(param, dens)
        ds[it] = calc_separation_distance(param, dens)
        σs[it] = sqrt(Ls[it]*Ls[it] - ds[it]*ds[it]/4)
        if save_anim
            plot(zs, dens.ρ; ylim=(0,0.3), xlabel="z [fm]", ylabel="ρ [fm⁻³]", label="ρ")
        end
    end

    p = plot(ts, Etots; xlabel="time [MeV⁻¹]", ylabel="total energy [MeV]", label=false)
    display(p)

    p = plot(ts, ds; xlabel="time [MeV⁻¹]", ylabel="separation distance [fm]", label=false)
    display(p)


    println("")
    @show E_fluc = abs(std(Etots)/mean(Etots))*100

    if save_anim
        gif(anim, "./1dimTDHF_figure/slab_collision.gif", fps = 15)
    end 
end










end # module
