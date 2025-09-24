"""
Parallel Search Execution for Batch Molecule Processing
"""

import os
import json
import time
import threading
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from llm_tree_optimizer import LLMGuidedTreeOptimizer
from data_loader import load_dataset_targets

# Thread-local storage for optimizer instances
thread_local_data = threading.local()


def get_thread_optimizer(args, inventory, template_dict, reaction_list, 
                        all_reaction_fps, datasub):
    """Get thread-local optimizer instance, create if doesn't exist"""
    if not hasattr(thread_local_data, 'optimizer'):
        print(f"[Thread-{threading.current_thread().ident}] Initializing optimizer...")
        thread_local_data.optimizer = LLMGuidedTreeOptimizer(
            args, inventory, template_dict, reaction_list, 
            all_reaction_fps, datasub
        )
    return thread_local_data.optimizer


def search_single_molecule(target_molecule: str, molecule_idx: int, 
                          total_molecules: int, log_dir: str, args,
                          inventory, template_dict, reaction_list,
                          all_reaction_fps, datasub, route_list, all_fps):
    """Search for synthesis route of a single molecule"""
    # Check if result already exists
    result_file = os.path.join(log_dir, f"result_{molecule_idx:05d}.json")
    if os.path.exists(result_file):
        print(f"[Skip] Molecule {molecule_idx+1}/{total_molecules} already completed, skipping...")
        with open(result_file, 'r') as f:
            return json.load(f)
    
    # Get thread-local optimizer
    optimizer = get_thread_optimizer(args, inventory, template_dict, 
                                    reaction_list, all_reaction_fps, datasub)
    optimizer.clear_cache()  # Clear cache from previous search
    
    start_time = time.time()
    thread_id = threading.current_thread().ident
    
    print(f"[Thread-{thread_id}] Starting search {molecule_idx+1}/{total_molecules}: "
          f"{target_molecule[:50]}...")
    
    try:
        # Search configuration
        config = {
            "max_iterations": 100,
            "max_depth": 16,
            "max_oracle_calls": args.max_oracle_calls
        }
        
        # Execute search
        solution = optimizer._optimize(target_molecule, route_list, all_fps, config)
        
        search_time = time.time() - start_time
        
        # Prepare result data
        is_complete_success = solution is not None and solution.get("type") != "partial_solution"
        is_partial_success = solution is not None and solution.get("type") == "partial_solution"

        result = {
            "molecule_idx": molecule_idx,
            "target_molecule": target_molecule,
            "success": is_complete_success,
            "partial_success": is_partial_success,
            "search_time": search_time,
            "iterations_completed": getattr(optimizer, 'current_iteration', 0),
            "total_and_nodes": optimizer.total_and_nodes,
            "solution": solution,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        }
        
        # Save individual result
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Determine status
        if is_complete_success:
            status = "✅ SUCCESS"
        elif is_partial_success:  
            status = "🔶 PARTIAL"
        else:
            status = "❌ FAILED"
            
        print(f"[Thread-{thread_id}] {status} {molecule_idx+1}/{total_molecules} "
              f"({search_time:.1f}s, {optimizer.total_and_nodes} nodes)")
        
        return result
        
    except Exception as e:
        search_time = time.time() - start_time
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"[Thread-{thread_id}] 💥 ERROR {molecule_idx+1}/{total_molecules}: {error_msg}")
        
        # Record error result
        result = {
            "molecule_idx": molecule_idx,
            "target_molecule": target_molecule,
            "success": False,
            "search_time": search_time,
            "error": error_msg,
            "error_trace": error_trace,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        }
        
        # Save error result
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result


def run_batch_search(args, inventory, template_dict, reaction_list,
                    all_reaction_fps, datasub, route_list, all_fps,
                    num_threads: int = 5, start_idx: int = 0, 
                    end_idx: Optional[int] = None):
    """Batch search for all molecules in dataset"""
    
    # Load target molecules
    print(f"Loading dataset: {args.dataset}")
    targets = load_dataset_targets(args.dataset)
    
    if end_idx is None:
        end_idx = len(targets)
    
    targets = targets[start_idx:end_idx]
    total_molecules = len(targets)
    
    print(f"Loaded {total_molecules} target molecules (index {start_idx} to {end_idx-1})")
    print(f"Using {num_threads} parallel threads")
    
    # Create log directory
    log_dir = f"./logs_v8/{args.dataset}/{args.api_model}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save search configuration
    config_info = {
        "dataset": args.dataset,
        "total_molecules": total_molecules,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_threads": num_threads,
        "start_time": datetime.now().isoformat(),
        "model_name": args.api_model,
        "max_oracle_calls": args.max_oracle_calls,
        "expansion": args.expansion
    }
    
    with open(os.path.join(log_dir, "search_config.json"), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # Execute parallel search
    start_time = time.time()
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                search_single_molecule, target, start_idx + i, total_molecules, 
                log_dir, args, inventory, template_dict, reaction_list,
                all_reaction_fps, datasub, route_list, all_fps
            ): i
            for i, target in enumerate(targets)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                eta = avg_time * (total_molecules - completed)
                print(f"\n🔄 Progress: {completed}/{total_molecules} "
                      f"({completed/total_molecules*100:.1f}%) "
                      f"ETA: {eta/60:.1f}min\n")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = sum(1 for r in results if r['success'])
    failed = total_molecules - successful
    success_rate = successful / total_molecules * 100 if total_molecules > 0 else 0
    
    # Save summary
    summary = {
        "dataset": args.dataset,
        "total_molecules": total_molecules,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "total_time": total_time,
        "avg_time_per_molecule": total_time / total_molecules,
        "num_threads": num_threads,
        "end_time": datetime.now().isoformat()
    }
    
    with open(os.path.join(log_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(log_dir, "all_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Batch search completed!")
    print(f"📊 Results: {successful}/{total_molecules} successful ({success_rate:.1f}%)")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"💾 Results saved to: {log_dir}")