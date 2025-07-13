# solver described here: https://www.robots.ox.ac.uk/~vgg/publications/2000/Schaffalitzky00/schaffalitzky00.pdf
using Random, LinearAlgebra
function skew(p::Vector{Float64}) # skew symmetric matrix from an image point
    [0 -p[3] p[2]; p[3] 0 -p[1]; -p[2] p[1] 0]
end
function hyp(p::Vector{Float64}) # "hypercamera" matrix from a world point
    [p[1] p[2] p[3] p[4] 0 0 0 0 0 0 0 0;
     0 0 0 0 p[1] p[2] p[3] p[4] 0 0 0 0;
     0 0 0 0 0 0 0 0 p[1] p[2] p[3] p[4]]
end
function hyp(p::Matrix{Float64})
    [p[1] p[2] p[3] p[4] 0 0 0 0 0 0 0 0;
     0 0 0 0 p[1] p[2] p[3] p[4] 0 0 0 0;
     0 0 0 0 0 0 0 0 p[1] p[2] p[3] p[4]]
end
function nullvectors(k::Int, M::Matrix{Float64}) # last singular vector... does svd algorithm matter?
    F = svd(M, full=true);
    n = size(F.V)[2];
    F.V[:,(n-k+1):n]
end
function sixpoint_SZHT(xs::Vector{Matrix{Float64}}) 
    # input: a vector of 3x6 matrices
    # output: (Xs, Ps): (at most 3) sets of 6 world pts, (at most 3) sets of 3 cameras
    # note: solution is valid up to world homography 
    # first five world points are columns of matrix E below
    E = hcat(I(4), ones(4));
    # step 0: extract camera pencils
    pencil_matrices = [vcat([skew(x[:,i]) * hyp(E[:,i]) for i=1:5]...) for x in xs];
    pencils = [nullvectors(2, pm) for pm in pencil_matrices];
    As = [reshape(pencil[:,1], 4, 3)' for pencil in pencils];
    Bs = [reshape(pencil[:,2], 4, 3)' for pencil in pencils];
    # step 1: write down 3x5 homogeneous system W * chi = 0
    Qs = [As[i]' * skew(xs[i][:, 6]) * Bs[i] - Bs[i]' * skew(xs[i][:, 6]) * As[i]  for i=1:3];
    W = vcat([[Q[1,2] Q[1,3] Q[2,3] Q[2,4] Q[3,4]] for Q in Qs]...);
    # step 2: extract nullspace of W.
    chi = nullvectors(2, W);
    # step 3: plug into cubic and solve
    lc_inverse = 1/(chi[1,1]*chi[2,1]*chi[4,1]-chi[2,1]*chi[3,1]*chi[4,1]-chi[1,1]*chi[2,1]*chi[5,1]+chi[1,1]*chi[3,1]*chi[5,1]-chi[1,1]*chi[4,1]*chi[5,1]+chi[2,1]*chi[4,1]*chi[5,1]);
    coeff_0 = lc_inverse * (chi[1,2]*chi[2,2]*chi[4,2]-chi[2,2]*chi[3,2]*chi[4,2]-chi[1,2]*chi[2,2]*chi[5,2]+chi[1,2]*chi[3,2]*chi[5,2]-chi[1,2]*chi[4,2]*chi[5,2]+chi[2,2]*chi[4,2]*chi[5,2]);
    coeff_1 = lc_inverse * (chi[4,1]*chi[1,2]*chi[2,2]-chi[5,1]*chi[1,2]*chi[2,2]+chi[5,1]*chi[1,2]*chi[3,2]-chi[4,1]*chi[2,2]*chi[3,2]+chi[2,1]*chi[1,2]*chi[4,2]-chi[5,1]*chi[1,2]*chi[4,2]+chi[1,1]*chi[2,2]*chi[4,2]-chi[3,1]*chi[2,2]*chi[4,2]+chi[5,1]*chi[2,2]*chi[4,2]-chi[2,1]*chi[3,2]*chi[4,2]-chi[2,1]*chi[1,2]*chi[5,2]+chi[3,1]*chi[1,2]*chi[5,2]-chi[4,1]*chi[1,2]*chi[5,2]-chi[1,1]*chi[2,2]*chi[5,2]+chi[4,1]*chi[2,2]*chi[5,2]+chi[1,1]*chi[3,2]*chi[5,2]-chi[1,1]*chi[4,2]*chi[5,2]+chi[2,1]*chi[4,2]*chi[5,2]);
    coeff_2 = lc_inverse * ((chi[2,1]*chi[4,1]*chi[1,2]-chi[2,1]*chi[5,1]*chi[1,2]+chi[3,1]*chi[5,1]*chi[1,2]-chi[4,1]*chi[5,1]*chi[1,2]+chi[1,1]*chi[4,1]*chi[2,2]-chi[3,1]*chi[4,1]*chi[2,2]-chi[1,1]*chi[5,1]*chi[2,2]+chi[4,1]*chi[5,1]*chi[2,2]-chi[2,1]*chi[4,1]*chi[3,2]+chi[1,1]*chi[5,1]*chi[3,2]+chi[1,1]*chi[2,1]*chi[4,2]-chi[2,1]*chi[3,1]*chi[4,2]-chi[1,1]*chi[5,1]*chi[4,2]+chi[2,1]*chi[5,1]*chi[4,2]-chi[1,1]*chi[2,1]*chi[5,2]+chi[1,1]*chi[3,1]*chi[5,2]-chi[1,1]*chi[4,1]*chi[5,2]+chi[2,1]*chi[4,1]*chi[5,2]));
    companion_matrix = [0 0 -coeff_0; 1 0 -coeff_1; 0 1 -coeff_2];
    # note: arbitrary tolerance for imaginary parts used below
    e_values = real.(filter(x -> abs(imag(x)) < 1e-6, eigvals(companion_matrix)));
    est_chis = [chi * [t; 1] for t in e_values];
    # the line below is for debugging: check cubic residuals are small
    # print("cubic residual small? ", all(h -> det([h[5] h[5] h[2]; h[4] h[3] h[2]; h[4] h[1] h[1]]) < 1e-8, est_chis), "\n");
    # step 4: recover world point with 
    design_matrices = [
        [c[5]-c[4]         0           0 c[1]-c[2]; 
         c[5]-c[3]         0        c[1]         0; 
         c[4]-c[3]      c[2]           0         0; 
                 0 c[5]-c[2] c[1] - c[4]         0; 
                 0      c[5]           0 c[1]-c[3]; 
                 0         0        c[4] c[2]-c[3]]
        for c in est_chis];
    recovered_X6s = [nullvectors(1, design_matrix) for design_matrix in design_matrices];
    # the line below is for debugging: check the estimated X6s lie on the appropriate quadric
    # print("recoverd X6s all on conic? ", all(y -> y < 1e-8, [norm(x' * Q * x) for x in recovered_X6s, Q in Qs]), "\n");
    # ... and the forgotten step 5... recover cameras
    cameras = [[reshape(nullvectors(1, vcat(pencil_matrices[i], skew(xs[i][:,6]) * hyp(rec))), 4, 3)' for i=1:3] for rec in recovered_X6s]
    worlds = [hcat(E, rec) for rec in recovered_X6s]
    (worlds, cameras)
end
function reproj(xs, X_est, cams_est)
    ps = [c * X_est for c in cams_est];
    ps_affine = [p[1:2,:] * inv(diagm(p[3,:])) for p in ps];
    xs_affine = [x[1:2,:] * inv(diagm(x[3,:])) for x in xs];
    norm(xs_affine - ps_affine)
end
function best_solution(xs, worlds, cameras)
    nsols = length(worlds);
    i_opt = findmin(i -> reproj(xs, worlds[i], cameras[i]), 1:nsols)[2];
    (worlds[i_opt], cameras[i_opt])
end

# example 1: 3D data using standard reference frame in P^3
E = hcat(I(4), ones(4))
X6 = randn(4, 1)
X = hcat(E, X6) # six world points to be recovered
Ps = [randn(3, 4) for i=1:3] # cameras
xs = [P*X for P in Ps]
# noiseless
(worlds, cameras) = sixpoint_SZHT(xs);
(X_best, cams_best) = best_solution(xs, worlds, cameras);
reproj(xs, X_best, cams_best)
# noisy: how stable is the solution?
s = .001 # std deviation of Gaussian noise in images
xs_noisy = [P*X + diagm([s,s,0]) * randn(3,6) for P in Ps]
(worlds, cameras) = sixpoint_SZHT(xs_noisy);
(X_best, cams_best) = best_solution(xs_noisy, worlds, cameras);
reproj(xs, X_best, cams_best)

# example 2: random data
X = randn(4, 6) # six world points to be recovered
Ps = [randn(3, 4) for i=1:3] # cameras
xs = [P*X + diagm([s,s,0]) * randn(3,6) for P in Ps]
# noiseless
(worlds, cameras) = sixpoint_SZHT(xs);
(X_best, cams_best) = best_solution(xs, worlds, cameras);
reproj(xs, X_best, cams_best)
# noisy: how stable is the solution?
s = .001 # std deviation of Gaussian noise in images
xs_noisy = [P*X + diagm([s,s,0]) * randn(3,6) for P in Ps]
(worlds, cameras) = sixpoint_SZHT(xs_noisy);
(X_best, cams_best) = best_solution(xs_noisy, worlds, cameras);
reproj(xs, X_best, cams_best)