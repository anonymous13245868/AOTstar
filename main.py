"""
Main Entry Point for LLM-Guided Tree Search Retrosynthesis
"""

import argparse
import os
import json
from pathlib import Path

from data_loader import (
    load_inventory, 
    load_training_data, 
    initialize_test_inventory
)
from parallel_search import search_single_molecule, run_batch_search

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-Guided Tree Search for Retrosynthesis'
    )
    
    # Dataset and model parameters
    parser.add_argument(
        '--dataset', type=str, default='pistachio_hard',
        choices=['pistachio_hard', 'pistachio_reachable', 
                  'uspto_190', 'uspto_easy'],
        help='Dataset to search'
    )
    
    # API model selection
    parser.add_argument(
        '--api_model', type=str, default='deepseek',
        help='API model to use for LLM queries (e.g., deepseek, gpt-4o-2024-11-20)'
    )
    parser.add_argument(
        '--api_temperature', type=float, default=0.7,
        help='Temperature for API model'
    )
    
    # Parallel and search range parameters
    parser.add_argument(
        '--threads', type=int, default=5, 
        help='Number of parallel threads'
    )
    parser.add_argument(
        '--start_idx', type=int, default=0, 
        help='Start index for molecules'
    )
    parser.add_argument(
        '--end_idx', type=int, default=None, 
        help='End index for molecules (None for all)'
    )
    parser.add_argument(
        '--single_test', type=str, default=None, 
        help='Test single molecule SMILES'
    )
    
    # File paths and directories
    parser.add_argument(
        '--template_path', type=str, 
        default='./dataset/idx2template_retro.json',
        help='Path to template dictionary'
    )
    parser.add_argument(
        '--inventory_path', type=str, 
        default='./dataset/inventory.pkl',
        help='Path to inventory file'
    )
    parser.add_argument(
        '--rule_based_set_path', type=str, 
        default='/PATH/TO/scscore/data/data_processed.csv',
        help='Path to rule-based set'
    )
    
    # Search parameters
    parser.add_argument(
        '--max_oracle_calls', type=int, default=1000,
        help='Maximum oracle calls'
    )
    parser.add_argument(
        '--freq_log', type=int, default=100,
        help='Logging frequency'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Output directory'
    )
    parser.add_argument(
        '--log_results', action='store_true', default=True,
        help='Whether to log results'
    )
    parser.add_argument(
        '--expansion', type=int, default=1,
        help='Number of expansion routes'
    )
    
    # Test mode
    parser.add_argument(
        '--test_mode', action='store_true', 
        help='Use small test inventory for debugging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("🚀 LLM-Guided Tree Search Starting...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.api_model}")
    print(f"Temperature: {args.api_temperature}")
    print(f"Threads: {args.threads}")
    print(f"Test mode: {args.test_mode}")
    
    # Load data
    print("\nLoading data...")
    
    # Load inventory
    if args.test_mode:
        print("Using test inventory...")
        inventory = initialize_test_inventory()
    else:
        print(f"Loading inventory from {args.inventory_path}...")
        inventory = load_inventory(args.inventory_path)
    
    # Load training data
    print("Loading training data...")
    (target_list, route_list, test_list, test_route_list, 
     test_hard_list, test_hard_route_list, all_fps, 
     reaction_list, all_reaction_fps, datasub, template_dict) = load_training_data()
    
    print(f"Loaded {len(route_list)} training routes")
    print(f"Loaded {len(template_dict)} templates")
    print("All data loaded successfully!")
    
    # Single molecule test
    if args.single_test:
        print(f"\nTesting single molecule: {args.single_test}")
        log_dir = "./logs_single_test"
        os.makedirs(log_dir, exist_ok=True)
        
        result = search_single_molecule(
            args.single_test, 0, 1, log_dir, args,
            inventory, template_dict, reaction_list,
            all_reaction_fps, datasub, route_list, all_fps
        )
        
        print(f"Result: {'Success' if result['success'] else 'Failed'}")
        if result.get('solution'):
            print(f"Solution type: {result['solution'].get('type')}")
    
    # Batch search
    else:
        run_batch_search(
            args=args,
            inventory=inventory,
            template_dict=template_dict,
            reaction_list=reaction_list,
            all_reaction_fps=all_reaction_fps,
            datasub=datasub,
            route_list=route_list,
            all_fps=all_fps,
            num_threads=args.threads,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )


if __name__ == "__main__":
    main()