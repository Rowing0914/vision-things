# # Q1
# using HomotopyContinuation
# @var x, y

# F = [-x^5 + x^4 + y - 2, x^2 + (y-1)^2 - 1]
# S = solutions(solve(System(F)))

# for i in 1:(length(S) - 1)
#     println(S[i])
# end

# Q2
using HomotopyContinuation
@var x, y, z
f = x^2 + y^2 + z^2 - 1
g1 = x - y
g2 = 2*x - z
h1 = x - z
h2 = y - z
h3 = x^2 - z
F = [f*g1*h1, f*g1*h2, f*g1*h3, f*g2*h1, f*g2*h2, f*g2*h3]
N = numerical_irreducible_decomposition(System(F))
print(N)