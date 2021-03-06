function calculate_pi(NBIN::Int= 1000000000)::Float64
    step = 1/NBIN
    sum = 0
    for i=1:NBIN
        x = (i+0.5)*step;
        sum += 4 / (1 + x^2)
    end
    pi = sum*step
    return pi
end

if length(ARGS)>0
    @time @show calculate_pi(parse(Int, ARGS[1]))
else
    @time @show calculate_pi()
end
