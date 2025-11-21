run_sim_only:
python experiments/sim/run_swap_only.py --out data/raw/sim_swap_only.csv --m 8 16 32 --E 4 8 16 --shots 512 1024 4096 --reps 5 --seed 123
python experiments/sim/run_swap_extended.py --out data/raw/sim_swap_extended.csv --m 64 128 --E 8 16 --shots 2048 4096 --reps 5 --seed 123 --dim 256