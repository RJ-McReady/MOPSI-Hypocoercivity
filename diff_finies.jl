using LinearAlgebra
using Statistics
using SparseArrays
using ProgressMeter
using Plots

function periodic_filter(index_vect::Vector{Int},dim_vect::Vector{Int})::Nothing
    """Indexes are updated periodically so that overflow and underflow situations are understood with respect to the tensors' dimensions"""
    for i in 1:length(index_vect)
        index_vect[i] = mod(index_vect[i]-1, dim_vect[i]) + 1
    end
    nothing
end

#Hash access function
function hash_function(index_vect::Vector{Int},dim_vect::Vector{Int},periodic_flag = true)::Int
    """Hash function to flatten tensors length(dim_vect)-tensors"""
    @assert length(index_vect) == length(dim_vect)
    my_index_vect = copy(index_vect)
    if periodic_flag
        periodic_filter(my_index_vect, dim_vect)
    end
    l = length(my_index_vect)
    s = 0
    for i in 1:l
        s = (my_index_vect[i] - 1) + s * dim_vect[i]
    end
    return s + 1 #Problem?
end


function hash_function(index_vect::Vector{Int},dim::Int)::Int
    """Hash function to flatten tensors length(dim_vect)-tensors, square case"""
    return hash_function(index_vect, dim * ones(Int,length(index_vect)))
end

function hash_function(i::Int,j::Int,dim_vect::Vector{Int})::Int
    """Hash function to flatten tensors length(dim_vect)-tensors, dimension 2 case"""
    return j + dim_vect[2]*i
end


#Terrain building, centered around 0
function vector_dimension(dim_vect::Vector{Int})::Int
    """Total dimension of tensor of a given dimension as a column vector."""
    dim = 1
    for i in dim_vect
        dim *= i
    end
    return dim
end

function multidimensional_indexes(dim_vect::Vector{Int})::Vector{Vector{Int}}
    """Supposed to get list of indexes possible given a dimensions' vector."""
    multi = []
    if length(dim_vect) == 1
        for i in 1:dim_vect[1]
            append!(multi,[[i]])
        end
        return multi
    else
        pred = multidimensional_indexes(dim_vect[1:(length(dim_vect)-1)])
        for i in 1:dim_vect[length(dim_vect)]
            for pred_ind in pred
                acc = copy(pred_ind)
                append!(acc,[i])
                append!(multi,[acc])
            end
        end
        return multi
    end
end


function terrain_builder(dim_vect::Vector{Int}, step_vect::Vector{Float64}, corner_point::Vector{Float64})::Array{Vector{Float64}}
    """Builds a terrain vector of a given step vector in the shape of cube which inferior corner is given."""
    terrain_vector = Array{Vector{Float64}}(undef, vector_dimension(dim_vect))
    possible_indexes = multidimensional_indexes(dim_vect)
    for index_vect in possible_indexes
        terrain_vector[hash_function(index_vect,dim_vect)] = index_vect .* step_vect + corner_point
    end
    return terrain_vector
end

#Matrix building
function first_order_differentiation_rigidity(rigidity::AbstractArray,index_vect::Vector{Int},dim_vect::Vector{Int},diff_direction::Int,step_in_direction::Float64, prefactor = 1.::Float64)::Nothing
    """Computes a matrix corresponding to a first order derivation in a periodical context. Side-effect action."""
    index_tilt = zeros(Int, length(index_vect))
    index_tilt[diff_direction] = 1

    index_vect_plus = index_vect + index_tilt
    index_vect_minus = index_vect - index_tilt

    i = hash_function(index_vect,dim_vect)
    j1 = hash_function(index_vect_plus,dim_vect)
    j2 = hash_function(index_vect_minus,dim_vect)

    rigidity[i,j1] += prefactor * 1/(2step_in_direction)
    rigidity[i,j2] += prefactor * (- 1/(2step_in_direction))

    nothing
end

function second_order_differentiation_rigidity(rigidity::AbstractArray,index_vect::Vector{Int},dim_vect::Vector{Int},diff_direction::Int,step_in_direction::Float64, prefactor = 1.::Float64)::Nothing
    """Computes a matrix corresponding to a second order derivation in a periodical context. Side-effect action."""
    index_tilt = zeros(Int, length(index_vect))
    index_tilt[diff_direction] = 1

    index_vect_plus = index_vect + index_tilt
    index_vect_minus = index_vect - index_tilt

    i = hash_function(index_vect,dim_vect)
    j1 = hash_function(index_vect_plus,dim_vect)
    j2 = hash_function(index_vect_minus,dim_vect)
    
    rigidity[i,j1] += prefactor * 1/(step_in_direction*step_in_direction)
    rigidity[i,j2] += prefactor * 1/(step_in_direction*step_in_direction)
    rigidity[i,i] += prefactor * -2/(step_in_direction*step_in_direction)

    nothing
end

#Matrix composition
function identity_matrix(matrix_dimension::Int,sparse_flag = true::Bool)::AbstractMatrix
    if sparse_flag
        identity = spzeros(Float64, matrix_dimension, matrix_dimension)
        for i in 1:matrix_dimension
            identity[matrix_dimension,matrix_dimension] = 1
        end
        return identity
    else
        return Matrix(1.0I, matrix_dimension, matrix_dimension)
    end
end

function total_regidity(terrain::Vector{Vector{Float64}}, step_vect::Vector{Float64}, D::Vector{Float64},C::Matrix{Float64},dim_vect::Vector{Int})::AbstractArray
    """Computes the total rigidity of the PDE problem"""
    tensor_dim = vector_dimension(dim_vect)
    rigidity = zeros(Float64, tensor_dim, tensor_dim)
    index_list = multidimensional_indexes(dim_vect)
    @showprogress "Assemblage de la rigidité" for index_vect in index_list
        current_position = C * terrain[hash_function(index_vect,dim_vect)]
        for i in 1:length(dim_vect)
            second_order_differentiation_rigidity(rigidity,index_vect, dim_vect, i, step_vect[i],D[i])
            first_order_differentiation_rigidity(rigidity, index_vect, dim_vect, i, step_vect[i],current_position[i])
        end
    end
    return rigidity + tr(C) * identity_matrix(tensor_dim,false)
end

#Gradient caluculations
function tensor_gradient(U::Vector{Float64},dim_vect::Vector{Int},step_vect::Vector{Float64})
    """Computes gradient through finite differences."""
    index_list = multidimensional_indexes(dim_vect)
    gradU = [zeros(length(dim_vect)) for k in 1:length(U)]
    for index_vect in index_list
        i = hash_function(index_vect,dim_vect)
        grad_vector = gradU[i]
        for diff_direction in 1:length(dim_vect)
            index_tilt = zeros(Int, length(index_vect))
            index_tilt[diff_direction] = 1

            index_vect_plus = index_vect + index_tilt
            index_vect_minus = index_vect - index_tilt

            j1 = hash_function(index_vect_plus,dim_vect)
            j2 = hash_function(index_vect_minus,dim_vect)
            
            grad_vector[diff_direction] = (U[j1] - U[j2])/(step_vect[diff_direction])
        end
        gradU[i] = grad_vector
    end
    return gradU
end

function tensor_gradient(U::Vector{Float64},dim_vect::Vector{Int},step::Float64)
    """Computes gradient through finite differences."""
    step_vect = step * ones(Float64,length(dim_vect))
    return tensor_gradient(U,dim_vect,step_vect)
end

#P - norms
function custom_norm_squared(vector::Vector{Float64}, matrix::Matrix{Float64})::Float64
    """Computes tXMX for given X and M."""
    return transpose(vector) * matrix * vector
end

#Derichlet Setup
function is_extremal(index_vect::Vector{Int},dim_vect::Vector{Int})::Bool
    """Checks if coordinate vector cooresponds to boundary point."""
    d = length(index_vect)
    @assert d == length(dim_vect)
    if d == 1
        return (index_vect[1] == 1)||(index_vect[1] == dim_vect[1])
    else
        return (index_vect[1] == 1)||(index_vect[1] == dim_vect[1]) || is_extremal(index_vect[2:d],dim_vect[2:d])
    end
end

function derichlet_reset(U::Vector{Float64},dim_vect::Vector{Int})::Nothing
    """Sets boundary points to 0 to enforce Derichlet conditions for the PDE. Side effect action."""
    index_list = multidimensional_indexes(dim_vect)
    for index_vect in index_list
        if is_extremal(index_vect,dim_vect)
            U[hash_function(index_vect,dim_vect)] = 0
        end
    end
    nothing
end

#show stuff
function show_2D_case(vector::Vector{Float64}, dim_vect::Vector{Int})
    """ Displays heatmap of function. """
    mat = reshape(vector,dim_vect[1],dim_vect[2])
    xgrid = collect(1:dim_vect[1])
    vgrid = collect(1:dim_vect[2])
    p = heatmap(xgrid,vgrid,mat)
end

#Initialisation
function terrain_parameters_initialisation(lattice_parameter::Vector{Float64},dim_vect::Vector{Int})
    """ Returns a terrain vector and the spacewise stepvector."""
    step_vect = lattice_parameter ./ dim_vect
    corner_point = - lattice_parameter /2
    return (terrain_builder(dim_vect,step_vect,corner_point), step_vect)
end

function terrain_parameters_initialisation(L::Float64, dim::Int, d::Int)
    """ Returns a terrain vector and the spacewise stepvector, in the case of a square terrain.
    Dim is the number of steps and d the number of physical dimensions of the PDE."""
    lattice_parameter = L* ones(Float64,d)
    dim_vect = dim * ones(Int,d)
    return terrain_parameters_initialisation(lattice_parameter,dim_vect)
end

function gaussian(entry_vector::Vector{Float64}, mean = 0.::Float64)::Float64
    mean_v = mean * ones(Float64,length(entry_vector))
    return exp(-(norm(entry_vector- mean_v)^2)/2)
end

function gaussian(entry_vector::Vector{Float64}, mean::Vector{Float64})::Float64
    return gaussian(entry_vector - mean)
end


#Special Handling
get_last_column(my_array) = my_array[:,length(my_array[1,:])]

maxa(array::AbstractArray) = reduce((x,y) -> max(x,y), array)

min(array::AbstractArray) = reduce((x,y) -> min(x,y), array)


function solve()
    n = 1
    d = 2n
    C = [0. 0. ; 0. 0.]
    D = [1.,1.]

    L = 15.
    dim = 100
    
    lattice_parameter = L* ones(Float64,d)
    dim_vect = dim * ones(Int,d)
    terrain, step_vect = terrain_parameters_initialisation(L, dim, d)
    println("Terrain done")

    U = gaussian.(terrain)
    rigidity = total_regidity(terrain, step_vect,D,C,dim_vect)
    println("Rigidity done")

    #######
    T = 1.
    J = 25
    time_step = T/J

    K = inv( Matrix(1.0I, vector_dimension(dim_vect), vector_dimension(dim_vect)) - time_step * rigidity )
    println("Rigidity inverted")

    U_tab = copy(U)
    @gif for j in 1:J
        if j%5 == 0
            @show j
        end
        show_2D_case(get_last_column(U_tab),dim_vect)
        U_new = K * get_last_column(U_tab)
        U_tab = [U_tab U_new]
    end
end

function debug_diff()
    n = 1
    d = 2n
    C = [0. 0. ; 0. 0.]
    D = [1.,1.]

    L = 15.
    dim = 3
    
    lattice_parameter = L* ones(Float64,d)
    dim_vect = dim * ones(Int,d)
    terrain, step_vect = terrain_parameters_initialisation(L, dim, d)
    println("Terrain done")

    U = gaussian.(terrain)
    rigidity = total_regidity(terrain, step_vect,D,C,dim_vect)
    println("Rigidity done")

    println("Sum of coefficients = $( sum(rigidity) )")
    println("Matrix is symmetric? $( issymmetric(rigidity) )")
end