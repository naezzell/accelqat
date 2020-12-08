println("========loading and compiling========")
include("try_gpu_accel.jl")

DEFAULT_N = 8
DEFAULT_TF = 32
println("\n========start testing========")

println("\n--------scaling n on cpu--------")
i=2
while i <= 10
    println("n=$(i); tf=$(DEFAULT_TF)")
    for t in 1:3
        r = nothing; GC.gc(true); CUDA.reclaim();
        @time r = anneal_spin_glass(i, DEFAULT_TF);
    end
    if i >= 8
        global i+=1;
    else
        global i+=2;
    end
end

println("\n--------scaling tf on cpu--------")
tf=4
while tf<=128
    println("n=$(DEFAULT_N); tf=$(tf)")
    for t in 1:3
        r = nothing; GC.gc(true); CUDA.reclaim();
        @time r = anneal_spin_glass(DEFAULT_N, tf);
    end
    global tf*=2;
end

println("\n--------scaling n on gpu--------")
i=2
while i<=10
    println("n=$(i); tf=$(DEFAULT_TF)")
    for t in 1:3
        r = nothing; GC.gc(true); CUDA.reclaim();
        @time r = anneal_spin_glass_gpu(i, DEFAULT_TF);
    end
    if i >= 8
        global i+=1;
    else
        global i+=2;
    end
end

println("\n--------scaling tf on cpu--------")
tf=4
while tf<=128
    println("n=$(DEFAULT_N); tf=$(tf)")
    for t in 1:3
        r = nothing; GC.gc(true); CUDA.reclaim();
        @time r = anneal_spin_glass(DEFAULT_N, tf);
    end
    global tf*=2;
end
