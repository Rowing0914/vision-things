using HomotopyContinuation
using Plots

@var x,t


f = x^5+13*x-1

S = solutions(solve(System([f])))

SCATTER = scatter([real(s) for s in S],[imag(s) for s in S])


#Takes one newton step starting at x0 w.r.t. poly f(x)
function newton(x0,f,x)
   return(x0-evaluate(f,x=>x0)/evaluate(differentiate(f,x),x=>x0))
end


####Let's animate a newton sequence

NewtonSequence = [randn(ComplexF64)]

for i in 1:5
	push!(NewtonSequence,newton(last(NewtonSequence),f,x))
end
println(NewtonSequence)
exit 0


function make_animation(PointMatrix; fps=15)
	anim = @animate for t in eachrow(PointMatrix)
	   scatter([real(p) for p in t],[imag(p) for p in t],xlims=[-4,4],ylims=[-4,4],legend=false)
	   scatter!([real(s) for s in S],[imag(s) for s in S])
	end
	gif(anim; fps=fps)
end

make_animation(NewtonSequence)





####Let's construct a homotopy

#Bezout start system
function bezout(f,x)
	return(x^degree(f)-1.0)
end

bezout(f,x)

function bezout_start(f)
	d = degree(f)
	return([exp(2*pi*im*k/d) for k in 1:d])
end

bezout_start(f)

#Bezout homotopy (with gamma trick)
function bezout_homotopy(f,x,t; gamma = (1+im)/sqrt(2))
	g = bezout(f,x)
	h = (1-t)*f+gamma*t*g
	return(h)
end

H = bezout_homotopy(f,x,t)


####Let's write the diffeq satisfied by the solution paths and apply Euler

#The davidenko differential equation for the homotopy paths of H(x;t)
function davidenko(H,x,t)
	-differentiate(H,t)/differentiate(H,x)
end

dav = davidenko(H,x,t)

function euler_step(start,diffeq,step,vars)
	(xi,ti) = start
	(x,t) = vars
	println("--")
	println(xi)
	println(ti)
	println(x)
	println(t)
	newx = xi+step*evaluate(diffeq,x=>xi,t=>ti)
	newt = ti+step
	return((newx,newt))
end

euler_step((bezout_start(f)[5],1),dav,-0.1,(x,t))

EulerSequence = [(bezout_start(f)[2],1.0)]

for i in 1:10
	push!(EulerSequence,euler_step(last(EulerSequence),dav,-0.1,(x,t)))
end


make_animation([e[1] for e in EulerSequence]; fps=10)


EulerSequences = [[(bezout_start(f)[i],1.0) for i in 1:5]]

for i in 1:10
push!(EulerSequences,[euler_step(last(EulerSequences)[i],dav,-0.1,(x,t)) for i in 1:5])
end

PointMatrix = reshape([EulerSequences[i][j][1] for j in 1:5 for i in 1:10 ],10,5)

make_animation(PointMatrix)




function predict_correct(current_sol,diffeq,step,H,vars)
	(xi,ti) = current_sol
	(x,t) = vars
	#predict
	(xeuler,teuler) = euler_step(current_sol,diffeq,step,vars)
	println("       Euler prediction: ",xeuler, " at ",teuler)
	#correct
	xnewton = newton(xeuler,evaluate(H,t=>teuler),x)
	println("       Newton correction: ",xnewton, " at ",teuler)
	return((xnewton,teuler))
end

PCSequence = [(bezout_start(f)[5],1.0)]

#Go from 10 and 0.1 to 100 and 0.01 to see the path crossing
for i in 1:10
	push!(PCSequence,predict_correct(last(PCSequence),dav,-0.1,H,(x,t)))
end

make_animation([p[1] for p in PCSequence])



function track_solution(start_sol,H,vars; stepsize=-0.1)
	println("Tracking start solution:", start_sol)
	(x,t) = vars
	current_sol = (start_sol,1)
	deq = davidenko(H,x,t)
	path = [start_sol]
	while current_sol[2]>0.0
		step = -min(abs(stepsize),current_sol[2])
		current_sol = predict_correct(current_sol,deq,step,H,vars)
		push!(path,current_sol[1])
	end
	return(path)
end

HomotopyPaths = [track_solution(b,H,(x,t)) for b in bezout_start(f)]
make_animation(hcat(HomotopyPaths...))

HomotopyPaths = [track_solution(b,H,(x,t);stepsize=-0.01) for b in bezout_start(f)]
make_animation(hcat(HomotopyPaths...))
