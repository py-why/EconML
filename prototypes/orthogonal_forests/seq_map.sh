dir=$1
tefunc=$2
outdir="${1}_postprocess_results"
mkdir -p $dir
mkdir -p $outdir
echo $dir
echo $outdir
# Monte Carlo parameters
n_dim=500
n_samples=5000
n_experiments=100
n_x=1
n_trees=100
bootstrap=0
subsample_power=0.88
min_leaf_size=5
max_splits=20
it=0
for tuples in 0,'--R' 1,'' 3,'' 4,''; 
do 
    IFS=',' read method_id run_R <<< "${tuples}"
    for support_size in 5 10;
    do  
        echo "Method id: ${method_id}"          
        echo $run_R >> "${dir}/log_${it}.txt"
        echo $it >> "${dir}/overall_log.txt"
        echo "Iteration ${it}" &>> "${dir}/log_${it}.txt"
        python monte_carlo.py --output_dir $dir --n_experiments $n_experiments --n_dim $n_dim --support_size $support_size --n_x $n_x --n_samples $n_samples --te_func $2 --max_splits $max_splits --min_leaf_size $min_leaf_size --n_trees $n_trees --bootstrap $bootstrap --subsample_power $subsample_power --method_id $method_id $run_R &>> "${dir}/log_${it}.txt"
        it=$((it+1))
    done
done

python comparison_plots.py --output_dir $outdir --input_dir $dir -merge &>> "${outdir}/log_${it}.txt"
echo "DONE"