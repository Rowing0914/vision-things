using HomotopyContinuation
using Plots

#Takes one newton step starting at x0 w.r.t. poly f(x)
function newton(x0,f,x)
   return(x0-evaluate(f,x=>x0)/evaluate(differentiate(f,x),x=>x0))
end


function make_animation(PointMatrix; fps=15)
	frame_rows = collect(eachrow(PointMatrix))
	for i in 1:10
		push!(frame_rows, last(frame_rows))
	end

	anim = @animate for t in frame_rows
	   scatter([real(p) for p in t],[imag(p) for p in t],xlims=[-4,4],ylims=[-4,4],legend=false)
	   scatter!([real(s) for s in S],[imag(s) for s in S])
	end
	gif(anim; fps=fps)
end

#Bezout start system
function bezout(f,x)
	return(x^degree(f)-1.0)
end

function bezout_start(f)
	d = degree(f)
	return([exp(2*pi*im*k/d) for k in 1:d])
end

#Bezout homotopy (with gamma trick)
function bezout_homotopy(f,x,t; gamma = (1+im)/sqrt(2))
	g = bezout(f,x)
	h = (1-t)*f+gamma*t*g
	return(h)
end

#The davidenko differential equation for the homotopy paths of H(x;t)
function davidenko(H,x,t)
	-differentiate(H,t)/differentiate(H,x)
end

function euler_step(start,diffeq,step,vars)
	(xi,ti) = start
	(x,t) = vars
	newx = xi+step*evaluate(diffeq,x=>xi,t=>ti)
	newt = ti+step
	return((newx,newt))
end

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

function track_solution(start_sol,H,vars; stepsize=-0.1, stopat=0.0)
	println("Tracking start solution:", start_sol)
	(x,t) = vars
	current_sol = (start_sol,1)
	deq = davidenko(H,x,t)
	println("Homotopy: ", H)
	println("Davidenko: ", deq)
	path = [start_sol]
	while current_sol[2]>stopat
		step = -min(abs(stepsize),current_sol[2]-stopat)
		current_sol = predict_correct(current_sol,deq,step,H,vars)
		push!(path,current_sol[1])
	end
	println("Final t-value: ", current_sol[2])
	return(path)
end

function mysolve(f, x; stepsize=-0.1, stopat=0.0, gamma=(1+im)/2)
	g = bezout(f, x)
	S = bezout_start(f)
	@var t
	H = bezout_homotopy(f, x, t; gamma=gamma)
	println(H)
	SolutionPaths = [track_solution(s, H, (x, t); stepsize=stepsize, stopat=stopat) for s in S]
	sols = [(last(SP), stopat) for SP in SolutionPaths]
end

function straight_line_track(start_sol, f, g, x; stepsize=-0.1, stopat=0.0, returnpath=false)
	println("Tracking start solution: ", start_sol)
	@var t
	current_sol = (start_sol, 1.0)
	H = (1 - t)*f + t*g
	deq = davidenko(H, x, t)
	println("Homotopy: ", H)
	println("Davidenko: ", deq)
	path = [start_sol]
	while current_sol[2] > stopat
		step = -min(abs(stepsize), current_sol[2]-stopat)
		current_sol = predict_correct(current_sol, deq, step, H, (x, t))
		push!(path, current_sol[1])
	end
	print("Final t-value: ", current_sol[2])
	if returnpath == true
		return(path)
	else
		return(last(path))
	end
end

function endgame(sol, H, x, t, N)
	println("Start endgame at ", sol)
	(xi, e) = sol
	single_circle = [e * exp(2 * pi * im * k/N) for k in 0: N]
	endgame_path = [xi]
	cycle_number = 0
	while cycle_number < 100 && (length(endgame_path) == 1 || abs(last(endgame_path) - first(endgame_path)) > 0.00001)
		cycle_number = cycle_number + 1
		println("Cycle number is currently: ", cycle_number)
		for i in 1:N
			println("Tracking ", single_circle[i], " ----> ", single_circle[i+1])
			s = last(endgame_path)
			start_system = evaluate(H, t => single_circle[i])
			target_system = evaluate(H, t => single_circle[i+1])
			s = straight_line_track(last(endgame_path), target_system, start_system, x)
			push!(endgame_path, s)
		end
	end
	println("Computed cycle number: ", cycle_number)
	cauchy_guess = (1/(length(endgame_path) - 1)) * sum(endgame_path[1:end-1])
	return(endgame_path, cycle_number, cauchy_guess)
end
