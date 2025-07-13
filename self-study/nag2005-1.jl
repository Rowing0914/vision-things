using HomotopyContinuation
using LinearAlgebra
using Groebner

# Define polynomial variables
@polyvar x y t

# Define f, g, h
f = [x^2 + 4 * y^2 - 4, 2*y^2 - x]
g = [x^2 - 1, y^2 - 1]
h = t * f + (1 - t) * g
println(h)

# Compute derivative of each element of eh with respect to x
jh = [[differentiate(h[1], x), differentiate(h[2], x)] [differentiate(h[1], y), differentiate(h[2], y)]]
println(jh)
sys = [h[1], h[2], det(jh)]
println(sys)
gb = groebner(sys)
println(gb)
