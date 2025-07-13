# https://github.com/peterkovesi/ImageProjectiveGeometry.jl/blob/b078ab94ba2e561977bfcf5e72e41ebeb0f7f0d6/test/test_projective.jl#L13
using Test, Statistics, LinearAlgebra

tol = 1e-8

# https://github.com/peterkovesi/ImageProjectiveGeometry.jl/blob/b078ab94ba2e561977bfcf5e72e41ebeb0f7f0d6/src/transforms3d.jl#L78
function rotx(theta::Real)
    T = [ 1.0     0.0         0.0      0.0
          0.0  cos(theta) -sin(theta)  0.0
          0.0  sin(theta)  cos(theta)  0.0
          0.0     0.0         0.0      1.0 ]
end

function roty(theta::Real)
    T = [ cos(theta)  0.0  sin(theta)  0.0
              0.0     1.0      0.0     0.0
         -sin(theta)  0.0  cos(theta)  0.0
              0.0     0.0      0.0     1.0 ]
end

# https://github.com/peterkovesi/ImageProjectiveGeometry.jl/blob/b078ab94ba2e561977bfcf5e72e41ebeb0f7f0d6/src/projective.jl#L31
function makehomogeneous(x::Array{T,2}) where T <: Real
    return [x; ones(1,size(x,2))]
end

function cameraproject(P::Array, pt::Array) # Projection matrix version

    if size(P) != (3,4)
        error("Projection matrix must be 3x4")
    end

    if size(pt, 1) != 3
        error("Points must be in a 3xN array")
    end

    nPts = size(pt,2)
    xy = zeros(2,nPts)

    for i in 1:nPts
        # Dehomogenise by dividing x and y components by w: [q_x/q_w, q_y/q_w]
        s = P[3,1]*pt[1,i] + P[3,2]*pt[2,i] + P[3,3]*pt[3,i] + P[3,4]  # scaling factor
        xy[1,i] = (P[1,1]*pt[1,i] + P[1,2]*pt[2,i] + P[1,3]*pt[3,i] + P[1,4])/s
        xy[2,i] = (P[2,1]*pt[1,i] + P[2,2]*pt[2,i] + P[2,3]*pt[3,i] + P[2,4])/s
    end

    return xy
end

# https://github.com/peterkovesi/ImageProjectiveGeometry.jl/blob/b078ab94ba2e561977bfcf5e72e41ebeb0f7f0d6/src/projective.jl#L1139
function normalise2dpts(ptsa::Array{T1,2}) where T1 <: Real

    pts = copy(ptsa) # Copy because we alter ptsa (should be able to fix this)
    newp = zeros(size(pts))

    if size(pts,1) != 3
        error("pts must be 3xN")
    end

    # Find the indices of the points that are not at infinity
    finiteind = findall(abs.(pts[3,:]) .> eps())

    # Should this warning be made?
    if length(finiteind) != size(pts,2)
        @warn("Some points are at infinity")
    end

    # For the finite points ensure homogeneous coords have scale of 1
    pts[1,finiteind] = pts[1,finiteind]./pts[3,finiteind]
    pts[2,finiteind] = pts[2,finiteind]./pts[3,finiteind]
    pts[3,finiteind] .= 1.0

    c = mean(pts[1:2,finiteind],dims=2)          # Centroid of finite points
    newp[1,finiteind] = pts[1,finiteind] .- c[1] # Shift origin to centroid.
    newp[2,finiteind] = pts[2,finiteind] .- c[2]

    dist = sqrt.(newp[1,finiteind].^2 .+ newp[2,finiteind].^2)
    meandist = mean(dist)

    scale = sqrt(2.0)/meandist

    T = [scale   0.  -scale*c[1]
         0.    scale -scale*c[2]
         0.      0.     1.      ]

    newpts = T*pts

    return newpts, T
end

function fundfromcameras(P1::Array{T1,2}, P2::Array{T2,2}) where {T1<:Real, T2<:Real}
    # Reference: Hartley and Zisserman p244
    # Version for projection matrices

    if (size(P1) != (3,4)) || (size(P2) != (3,4))
        error("Camera matrices must be 3x4")
    end

    C1 = nullspace(P1)  # Camera centre 1 is the null space of P1
    e2 = P2*C1          # epipole in camera 2

    e2x = [   0   -e2[3] e2[2]    # Skew symmetric matrix from e2
            e2[3]    0  -e2[1]
           -e2[2]  e2[1]   0  ]

    return F = e2x*P2*pinv(P1)
end

# Generate a set of 3D points
pts = rand(3,12)

# Intrinsic parameters
f = 4000
rows = 3000
cols = 4000
ppx = cols / 2
ppy = rows / 2
s = 0
K = [f s ppx;
     0 f ppy;
     0 0  1]

# Extrinsic parameters
X = 0
Y = 0
Z = 10
Rc_w = (rotx(pi) * roty(0.1))[1:3, 1:3]  # Rotation matrix
t = [X; Y; Z]  # Column vector

# Projection matrix; Camera1
P1 = K * hcat(Rc_w, t)
println(P1)

# Projection matrix; Camera2
t = [X+2; Y; Z]  # Column vector
Rc_w = (rotx(pi-.1)*roty(-0.1))[1:3,1:3]
P2 = K * hcat(Rc_w, t)
println(P2)

# Project points into images: Input of 8-pt algo
xy1 = cameraproject(P1, pts)
xy2 = cameraproject(P2, pts)

# ==== 8-pt algorithm: Estimate F from corresponding image points
(dim, npts) = size(xy1)

(x1, Tx1) = normalise2dpts(makehomogeneous(xy1))
(x2, Tx2) = normalise2dpts(makehomogeneous(xy2))
A = [x2[1:1,:]'.*x1[1:1,:]'   x2[1:1,:]'.*x1[2:2,:]'  x2[1:1,:]'  x2[2:2,:]'.*x1[1:1,:]'   x2[2:2,:]'.*x1[2:2,:]'  x2[2:2,:]'   x1[1:1,:]'   x1[2:2,:]'  ones(npts,1) ]
(U,D,V) = svd(A, full=true)

# Extract fundamental matrix from the column of V corresponding to smallest singular value.
F = reshape(V[:,9], 3, 3)'

# Enforce constraint that fundamental matrix has rank 2 by performing
# a svd and then reconstructing with the two largest singular values.
(U,D,V) = svd(F)
F = U*diagm(0 => [D[1], D[2], 0])*V'

# Denormalise
F1 = Tx2'*F*Tx1
# ==== 8-pt algorithm: Estimate F from corresponding image points

# Ground-trugh: Form fundamental matrix from the camera projection matrices
F2 = fundfromcameras(P1, P2)

# Adjust matrices to the same scale
F1 = F1/F1[3,3]
F2 = F2/F2[3,3]

@test maximum(abs.(F1 - F2)) < tol
