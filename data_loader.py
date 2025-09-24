"""
Data Loading Utilities for Retrosynthesis Search
"""

import pickle
import ast
from typing import List, Dict, Any


def load_inventory(inventory_path: str):
    """Load inventory from pickle file"""
    with open(inventory_path, 'rb') as file:
        inventory = pickle.load(file)
    return inventory


def load_dataset_targets(dataset_name: str) -> List[str]:
    """Load target molecules for specified dataset"""
    dataset_paths = {
        'pistachio_hard': "/PATH/TO/dataset/pistachio_hard_targets.txt",
        'pistachio_reachable': "/PATH/TO/dataset/pistachio_reachable_targets.txt",
        'pistachio_reachable_hard': "/PATH/TO/dataset/pistachio_reachable_hard.txt",
        'uspto_190': "/PATH/TO/dataset/uspto_190_targets.txt",
        'pistachio_easy': "/PATH/TO/dataset/pistachio_easy.txt"
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = dataset_paths[dataset_name]
    targets = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                tuple_data = ast.literal_eval(line.strip())
                targets.append(tuple_data[0])
    
    return targets


def load_training_data() -> tuple:
    """Load all training data from pickle files"""
    # Load route data
    train_data = '/PATH/TO/dataset/routes_train.pkl'
    val_data = '/PATH/TO/dataset/routes_val.pkl'
    test_data = '/PATH/TO/dataset/routes_test.pkl'
    test_hard_data = '/PATH/TO/dataset/routes_possible_test_hard.pkl'
    
    train_routes = pickle.load(open(train_data, 'rb'))
    val_routes = pickle.load(open(val_data, 'rb'))
    test_routes = pickle.load(open(test_data, 'rb'))
    test_hard_routes = pickle.load(open(test_hard_data, 'rb'))
    
    # Combine training and validation routes
    total_routes = train_routes + val_routes
    target_list = []
    route_list = []
    
    for route in total_routes:
        target_list.append(route[0].split('>>')[0])
        route_list.append(route)
    
    # Process test routes
    test_list = []
    test_route_list = []
    for route in test_routes:
        test_list.append(route[0].split('>>')[0])
        test_route_list.append(route)
    
    # Process hard test routes
    test_hard_list = []
    test_hard_route_list = []
    for route in test_hard_routes:
        test_hard_list.append(route[0].split('>>')[0])
        test_hard_route_list.append(route)
    
    # Load fingerprints and other data
    with open('./dataset/all_fps.pkl', 'rb') as f:
        all_fps = pickle.load(f)
    
    with open('./dataset/reaction_list.pkl', 'rb') as f:
        reaction_list = pickle.load(f)
    
    with open('./dataset/all_reaction_fps.pkl', 'rb') as f:
        all_reaction_fps = pickle.load(f)
    
    with open('./dataset/datasub.pkl', 'rb') as f:
        datasub = pickle.load(f)
    
    with open('./dataset/template_dict.pkl', 'rb') as f:
        template_dict = pickle.load(f)
    
    return (target_list, route_list, test_list, test_route_list, 
            test_hard_list, test_hard_route_list, all_fps, 
            reaction_list, all_reaction_fps, datasub, template_dict)


def initialize_test_inventory():
    """Initialize a test inventory for debugging"""
    from syntheseus.search.mol_inventory import SmilesListInventory
    
    test_smiles = [
        'CC', 'CCO', 'c1ccccc1', 'CC(=O)O', 'CN', 'C=O',
        'c1ccc(Cl)cc1', 'c1ccc(Br)cc1', 'c1ccc(O)cc1', 'c1cccnc1',
        'CCC', 'CCCO', 'c1ccc(N)cc1', 'c1ccc(C=O)cc1', 'CCN',
        'c1cnccn1', 'C1CCNCC1', 'CS(=O)(=O)Cl', 'CC(=O)Cl'
    ]
    
    return SmilesListInventory(test_smiles)