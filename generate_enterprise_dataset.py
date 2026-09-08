"""
HPC Parallel Structural Dataset Generator
Utilizes Python multiprocessing ProcessPoolExecutor across all CPU cores
to generate 5,000 multi-story multi-bay Eurocode FEA simulations into enterprise_fea_dataset.csv.
"""

import os
import time
import random
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from fea_test import MultiStoryFEAEngine
from train_model import train_and_save_surrogate_model

def run_single_simulation(item):
    """Worker function executing a single 3D multi-story structural FEA model."""
    run_id, n_stories, n_bays_x, n_bays_z, h, w_x, w_z, G_k, Q_k, W_k = item
    try:
        engine = MultiStoryFEAEngine(
            num_bays_x=n_bays_x,
            num_bays_z=n_bays_z,
            num_stories=n_stories,
            bay_width_x=w_x,
            bay_width_z=w_z,
            story_height=h,
            G_k=G_k,
            Q_k=Q_k,
            W_k=W_k
        )
        df, opt, _ = engine.build_and_analyze()

        return {
            'run_id': run_id,
            'num_stories': n_stories,
            'num_bays_x': n_bays_x,
            'num_bays_z': n_bays_z,
            'story_height': h,
            'bay_width_x': w_x,
            'bay_width_z': w_z,
            'G_k': G_k,
            'Q_k': Q_k,
            'W_k': W_k,
            'total_height_m': opt['total_height_m'],
            'delta_sway_mm': opt['delta_sway_mm'],
            'delta_sway_lim_mm': opt['delta_sway_lim_mm'],
            'util_sway': opt['util_sway'],
            'sway_status': opt['sway_status'],
            'beam_M_u_kNm': opt['max_beam_M'],
            'col_P_u_kN': opt['max_col_P'],
            'optimal_beam_sec': opt['optimal_beam']['name'],
            'optimal_col_sec': opt['optimal_col']['name'],
            'beam_mass_kg_m': opt['optimal_beam']['mass'],
            'col_mass_kg_m': opt['optimal_col']['mass'],
            'beam_util_pct': opt['optimal_beam']['util_pct'],
            'col_util_pct': opt['optimal_col']['util_pct'],
            'total_steel_weight_kg': opt['total_weight_kg']
        }
    except Exception as e:
        return None

def generate_hpc_enterprise_dataset(num_simulations=5000, csv_filename="enterprise_fea_dataset.csv"):
    """Runs parallel FEA simulations across all available CPU cores using ProcessPoolExecutor."""
    cpu_cores = os.cpu_count() or 4
    print("==================================================")
    print("      HPC PARALLEL FEA DATASET GENERATOR          ")
    print("==================================================")
    print(f"Target Simulations : {num_simulations:,}")
    print(f"Available CPU Cores: {cpu_cores}")
    print("==================================================\n")

    random.seed(42)
    work_items = []

    # Generate sampling parameter space
    for i in range(1, num_simulations + 1):
        n_stories = random.randint(1, 5)
        n_bays_x = random.randint(1, 3)
        n_bays_z = random.randint(1, 3)

        h = round(random.uniform(3.0, 4.5), 2)
        w_x = round(random.uniform(4.0, 8.0), 2)
        w_z = round(random.uniform(4.0, 8.0), 2)

        G_k = round(random.uniform(5.0, 30.0), 2)
        Q_k = round(random.uniform(2.0, 15.0), 2)
        W_k = round(random.uniform(0.0, 25.0), 2)

        work_items.append((i, n_stories, n_bays_x, n_bays_z, h, w_x, w_z, G_k, Q_k, W_k))

    start_time = time.time()
    results = []
    completed_count = 0

    # Parallel Execution with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = {executor.submit(run_single_simulation, item): item[0] for item in work_items}

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
            completed_count += 1
            if completed_count % 500 == 0 or completed_count == num_simulations:
                elapsed = time.time() - start_time
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed_count:,}/{num_simulations:,} ({completed_count / num_simulations * 100:.1f}%) | Speed: {rate:.1f} sim/sec | Elapsed: {elapsed:.1f}s")

    # Sort by run_id
    results.sort(key=lambda x: x['run_id'])

    df_dataset = pd.DataFrame(results)
    df_dataset.to_csv(csv_filename, index=False)
    
    # Also save as default fea_dataset.csv for backward compatibility
    df_dataset.to_csv("fea_dataset.csv", index=False)

    total_time = time.time() - start_time
    print("\n==================================================")
    print("          HPC GENERATION COMPLETE                 ")
    print("==================================================")
    print(f"Successfully saved {len(df_dataset):,} runs to '{csv_filename}'.")
    print(f"Total Execution Time : {total_time:.2f} seconds ({total_time/60.0:.2f} minutes)")
    print(f"Average Throughput   : {len(df_dataset)/total_time:.2f} simulations/sec")
    print("==================================================\n")

    # Train AI Surrogate Model on the new Enterprise HPC Dataset
    print("Retraining AI Neural Surrogate on Enterprise Dataset...")
    train_and_save_surrogate_model(csv_filename, "surrogate_model.pkl")

    return df_dataset

if __name__ == '__main__':
    generate_hpc_enterprise_dataset(5000, "enterprise_fea_dataset.csv")
