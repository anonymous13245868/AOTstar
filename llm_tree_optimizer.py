"""
LLM-Guided Tree Optimizer for Retrosynthesis Planning
"""

import re
import ast
import math
import time
from typing import List, Dict, Optional, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
from syntheseus import Molecule

from tree_nodes import ORNode, ANDNode
from prompts import construct_full_prompt
from utils import sanitize_smiles
from optimizer import BaseOptimizer, process_reaction_routes, extract_molecules_from_output


class LLMGuidedTreeOptimizer(BaseOptimizer):
    """LLM-guided optimizer based on AND-OR Tree"""
    
    def __init__(self, args=None, inventory=None, template_dict=None, 
                 reaction_list=None, all_reaction_fps=None, datasub=None):
        super().__init__(args, inventory, template_dict, reaction_list, 
                        all_reaction_fps, datasub)
        self.model_name = "llm_tree_planner"
        
        # API model configuration
        self.api_model = getattr(args, 'api_model', 'deepseek')
        self.api_temperature = getattr(args, 'api_temperature', 0.7)
        self.api_max_tokens = getattr(args, 'api_max_tokens', 4096)
        
        # AND-OR Tree state
        self.root_or_node: Optional[ORNode] = None
        self.all_molecules: Dict[str, ORNode] = {}  # smiles -> ORNode
        self.leaf_and_nodes: Set[ANDNode] = set()
        self.total_and_nodes = 0
        
        # Search parameters
        self.max_iterations = 100
        self.max_depth = 16
        self.expansion_routes = 1
        
        print(f"Initialized LLM-Guided Tree Optimizer")
    
    def clear_cache(self):
        """Clear search cache"""
        self.root_or_node = None
        self.all_molecules.clear()
        self.leaf_and_nodes.clear()
        self.total_and_nodes = 0
        print("Cleared search cache")
    
    def get_or_create_or_node(self, molecule_smiles: str) -> ORNode:
        """Get or create OR node"""
        molecule_smiles = sanitize_smiles(molecule_smiles)
        if molecule_smiles is None:
            raise ValueError(f"Invalid SMILES: {molecule_smiles}")
        
        if molecule_smiles not in self.all_molecules:
            # Check if molecule is purchasable
            is_solved = self.inventory.is_purchasable(Molecule(molecule_smiles))
            
            or_node = ORNode(
                molecule=molecule_smiles,
                smiles=molecule_smiles,
                is_solved=is_solved
            )
            self.all_molecules[molecule_smiles] = or_node
        
        return self.all_molecules[molecule_smiles]
    
    def _expand_root(self, route_list, all_fps):
        """Expand root node with retry mechanism"""
        retry_count = 0
        while retry_count < 3:
            try:
                retry_count += 1
                if retry_count > 1:
                    print(f"🔄 Retrying root expansion (attempt {retry_count})...")
                
                and_nodes = self._expand_or_node_with_llm(self.root_or_node, route_list, all_fps)
                if and_nodes:
                    print(f"✅ Root expansion succeeded on attempt {retry_count}")
                    return and_nodes
            except Exception as e:
                print(f"Root expansion error (attempt {retry_count}): {e}")
                continue

    def _optimize(self, target, route_list, all_fps, config):
        """Main optimization loop"""
        self.clear_cache()
        self.max_iterations = config.get("max_iterations", 100)
        self.max_depth = config.get("max_depth", 20)
        self.current_iteration = 0
        
        print(f"\n🔍 Starting LLM-Guided Tree Search:")
        print(f"   Target: {target}")
        
        # Initialize root node
        self.root_or_node = self.get_or_create_or_node(target)
        
        if self.root_or_node.is_solved:
            print("✅ Target molecule already purchasable!")
            return self._extract_trivial_solution()
        
        # Initial expansion with retry
        print("🔄 Starting initial expansion...")
        initial_and_nodes = self._expand_root(route_list, all_fps)
        print(f"Initially, generated {len(initial_and_nodes)} new AND nodes")
        
        # Initial evaluation
        for and_node in initial_and_nodes:
            score = self._evaluate_and_node_chemical(and_node)
            self._initialize_and_node_stats(and_node, score)
        
        # Main search loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            if iteration % 1 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Update global solution status
            self._update_global_solved_status()
            
            if self.root_or_node.is_solved:
                print(f"✅ Found complete solution at iteration {iteration + 1}")
                break
            
            # Selection: Choose AND node for expansion
            selected_and_node = self._selection_phase()
            if selected_and_node is None:
                print("No expandable AND node found")
                
                # Re-expand root if iterations remain
                if iteration < self.max_iterations - 1:
                    print(f"🔄 Re-expanding root node at iteration {iteration + 1}")
                    new_root_and_nodes = self._expand_root(route_list, all_fps)
                    print(f"   Generated {len(new_root_and_nodes)} new AND nodes from root re-expansion")
                    
                    for and_node in new_root_and_nodes:
                        score = self._evaluate_and_node_chemical(and_node)
                        self._initialize_and_node_stats(and_node, score)
                    continue
                else:
                    print("Search terminated: reached maximum iterations")
                    break
            
            # Expansion: Generate new routes for selected AND node
            new_and_nodes = self._expansion_phase(selected_and_node, route_list, all_fps)
            
            # Evaluation: Evaluate new nodes
            for new_and_node in new_and_nodes:
                score = self._evaluate_and_node_chemical(new_and_node)
                self._initialize_and_node_stats(new_and_node, score)
                self._backpropagate_and_node(new_and_node, score)
            
            # Check termination condition
            # if len(self.oracle) > config.get("max_oracle_calls", 100):
            #     print("Reached maximum oracle calls")
            #     break
        
        if self.root_or_node.is_solved:
            return self._extract_solution()
        else:
            return self._extract_partial_solution()
    
    def _selection_phase(self) -> Optional[ANDNode]:
        """Selection phase: Use UCB to select best AND node for expansion"""
        expandable_nodes = [
            node for node in self.leaf_and_nodes
            if (node.depth < self.max_depth and 
                not self._is_and_node_fully_solved(node) and
                self._has_expandable_reactants(node))
        ]
        
        if not expandable_nodes:
            return None
        
        # UCB1 selection
        best_node = max(expandable_nodes, key=lambda x: self._get_ucb_score(x))
        
        print(f"🎯 Selected AND node at depth {best_node.depth} "
              f"(UCB: {self._get_ucb_score(best_node):.3f}, visits: {best_node.visit_count})")
        
        return best_node
    
    def _expansion_phase(self, selected_and_node: ANDNode, route_list: List, all_fps: List) -> List[ANDNode]:
        """Expansion phase: Generate new child routes for selected AND node"""
        print(f"🔄 Expanding AND node at depth {selected_and_node.depth}")
        
        # Find unsolved reactants
        unsolved_reactants = [
            or_node for or_node in selected_and_node.child_molecules
            if not or_node.is_solved
        ]
        
        if not unsolved_reactants:
            print("   All reactants solved, no expansion needed")
            selected_and_node.is_leaf = False
            self.leaf_and_nodes.discard(selected_and_node)
            return []
        
        # Select expansion target
        target_reactant = self._select_expansion_target(unsolved_reactants)
        print(f"   Expanding reactant: {target_reactant.smiles}")
        
        # Root expansion
        new_and_nodes = []

        try:
            # Generate routes with LLM
            generated_routes = self._generate_routes_with_llm(target_reactant, route_list, all_fps)
            
            if generated_routes:
                # Map routes to tree structure
                new_and_nodes = self._map_routes_to_tree(
                    target_reactant, generated_routes, selected_and_node.depth + 1
                )
                
                if new_and_nodes:
                    print(f"   ✅ Expansion succeeded")
        except Exception as e:
            print(f"   Expansion error: {e}")

        if not new_and_nodes:
            print(f"   ❌ Expansion failed, marking node as exhausted")
            selected_and_node.is_leaf = False
            self.leaf_and_nodes.discard(selected_and_node)

        print(f"   Generated {len(new_and_nodes)} new AND nodes")
        return new_and_nodes
    
    def _generate_routes_with_llm(self, target_or_node: ORNode, route_list: List, all_fps: List) -> List[List[Dict]]:
        """Generate routes for target OR node using LLM"""
        # Get similar routes for reference (RAG)
        similar_routes = self._get_similar_routes(
            target_or_node.smiles, route_list, all_fps, num_examples=3
        )
        
        # Build examples
        examples = ''
        for route in similar_routes:
            examples += '<ROUTE>\n' + str(process_reaction_routes(route)) + '\n</ROUTE>\n'
        
        # Construct prompt
        question = construct_full_prompt(
            target_or_node.smiles, examples, self.expansion_routes
        )
        
        print(f'Query LLM agent using model: {self.api_model}...')
        message, answer = self.query_LLM(
            question, temperature=self.api_temperature, model=self.api_model
        )
        print('response...')
        print(answer)
        
        # Parse multiple routes
        return self._parse_multiple_routes(answer, target_or_node.smiles)
    
    def _parse_multiple_routes(self, llm_response: str, target_smiles: str) -> List[List[Dict]]:
        """Parse multiple routes from LLM response"""
        routes = []
        
        try:
            # Extract ROUTE section
            match = re.search(r'<ROUTE>(.*?)<ROUTE>', llm_response, re.DOTALL)
            if match == None:
                match = re.search(r'<ROUTE>(.*?)</ROUTE>', llm_response, re.DOTALL)
            if not match:
                print("No <ROUTES> section found in LLM response")
                return []
            
            routes_content = match.group(1)
            parsed_routes = ast.literal_eval(routes_content)
            if self.expansion_routes == 1:
                parsed_routes = [parsed_routes]
            
            for i, route in enumerate(parsed_routes):
                try:
                    # Clean route format
                    cleaned_route = self._clean_route(route, target_smiles)
                    if cleaned_route:
                        routes.append(cleaned_route)
                except Exception as e:
                    print(f"Failed to parse route {i}: {e}")
                    continue
            
        except Exception as e:
            print(f"Failed to parse multiple routes: {e}")
        
        print(f"Successfully parsed {len(routes)} routes from LLM response")
        return routes
    
    def _clean_route(self, route: List[Dict], target_smiles: str) -> Optional[List[Dict]]:
        """Clean single route"""
        try:
            # Check if last step is valid
            if len(route) >= 2:
                comp1 = ast.literal_eval(route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(route[-2]['Updated molecule set'])
                last_step_reactants = route[-1]['Reactants']
                
                if (set(comp1) == set(comp2) or 
                    last_step_reactants in ["", "[]", "None", "[None]"]):
                    route = route[:-1]
                    print('Route cleaned!')
            
            # Validate format
            for step in route:
                ast.literal_eval(step['Molecule set'])
                ast.literal_eval(step['Reaction'])
                ast.literal_eval(step['Reactants'])
                ast.literal_eval(step['Updated molecule set'])
            
            return route
            
        except Exception as e:
            print(f"Route cleaning failed: {e}")
            return None
    
    def _map_routes_to_tree(self, target_or_node: ORNode, routes: List[List[Dict]], 
                           base_depth: int) -> List[ANDNode]:
        """Map parsed routes to AND-OR tree structure"""
        new_and_nodes = []
        all_created_nodes = []
        
        for route_idx, route in enumerate(routes):
            try:
                # Validate route
                validated_route = self._validate_route_with_templates(route, target_or_node.smiles)
                if not validated_route:
                    continue
                
                # Create AND node for first step
                first_step = validated_route[0]
                and_node = self._create_and_node_from_step(
                    first_step, target_or_node, route_idx, base_depth
                )
                
                if and_node:
                    new_and_nodes.append(and_node)
                    all_created_nodes.append(and_node)
                    
                    # Recursively map remaining steps
                    remaining_nodes = self._map_remaining_steps(and_node, validated_route[1:])
                    all_created_nodes.extend(remaining_nodes)
                
            except Exception as e:
                print(f"Failed to map route {route_idx}: {e}")
                continue
        
        return all_created_nodes
    
    def _validate_route_with_templates(self, route: List[Dict], target_smiles: str) -> Optional[List[Dict]]:
        """Validate route using templates"""
        try:
            checked_route, final_evaluation = self.sanitize([target_smiles], route, exploration_signal=True)
            
            # Check if validation passed
            if self._is_route_valid(final_evaluation):
                return checked_route
            else:
                print(f"Route validation failed")
                return None
                
        except Exception as e:
            print(f"Route validation error: {e}")
            return None
    
    def _is_route_valid(self, evaluation: List) -> bool:
        """Check route validation result"""
        if not evaluation:
            return False
        
        # Check if first step is valid
        first_step = evaluation[0]
        if len(first_step) < 3 or not first_step[1]:
            return False
        
        return True
    
    def _create_and_node_from_step(self, step: Dict, parent_or_node: ORNode, 
                                  route_idx: int, depth: int) -> Optional[ANDNode]:
        """Create AND node from route step"""
        try:
            # Extract reaction info
            reaction = ast.literal_eval(step['Reaction'])[0]
            reactants_smiles = ast.literal_eval(step['Reactants'])
            product_smiles = parent_or_node.smiles
            
            # Create AND node
            reaction_id = f"{product_smiles}_{route_idx}_{depth}_{hash(str(reactants_smiles)) % 10000}"
            and_node = ANDNode(
                reaction_id=reaction_id,
                reactants=reactants_smiles,
                product=product_smiles,
                parent_molecule=parent_or_node,
                depth=depth
            )
            
            # Connect to parent OR node
            parent_or_node.child_reactions.append(and_node)
            
            # Create reactant OR nodes
            available_reactants = []
            for reactant_smiles in reactants_smiles:
                reactant_or_node = self.get_or_create_or_node(reactant_smiles)
                reactant_or_node.parent_reactions.append(and_node)
                and_node.child_molecules.append(reactant_or_node)
                
                if reactant_or_node.is_solved:
                    available_reactants.append("✅")
                else:
                    available_reactants.append("❌")

            # Print reactant availability
            availability_str = " ".join(available_reactants)
            print(f"   Reactants availability: {availability_str} "
                  f"({sum(1 for x in available_reactants if x == '✅')}/{len(available_reactants)})")
            
            self.total_and_nodes += 1
            self.leaf_and_nodes.add(and_node)
            
            # Print reaction info
            reactants_str = " + ".join([r[:30] + "..." if len(r) > 30 else r for r in reactants_smiles])
            print(f"🧪 New reaction: {product_smiles[:30]}... → {reactants_str}")
            
            return and_node
            
        except Exception as e:
            print(f"Failed to create AND node: {e}")
            return None
    
    def _map_remaining_steps(self, parent_and_node: ANDNode, remaining_steps: List[Dict]) -> List[ANDNode]:
        """Recursively map remaining route steps"""
        created_nodes = []
        
        if not remaining_steps:
            return created_nodes
        
        # Process steps in order
        current_step = remaining_steps[0]
        remaining_after_current = remaining_steps[1:]
        
        try:
            # Find product molecule for current step
            product = extract_molecules_from_output(current_step['Product'])[0]
            product_smiles = sanitize_smiles(product)
            
            # Find matching OR node in parent's child molecules
            target_or_node = None
            for child_or in parent_and_node.child_molecules:
                if child_or.smiles == product_smiles and not child_or.is_solved:
                    target_or_node = child_or
                    break
            
            if target_or_node:
                # Create new AND node for this OR node
                child_and_node = self._create_and_node_from_step(
                    current_step, target_or_node, 0, parent_and_node.depth + 1
                )
                
                if child_and_node:
                    created_nodes.append(child_and_node)
                    # Recursively process remaining steps
                    deeper_nodes = self._map_remaining_steps(child_and_node, remaining_after_current)
                    created_nodes.extend(deeper_nodes)
                    
        except Exception as e:
            print(f"Failed to map step: {e}")
        
        return created_nodes
    
    def _evaluate_and_node_chemical(self, and_node: ANDNode) -> float:
        """Improved chemical scoring with availability reward"""
        try:
            # Calculate availability score
            total_reactants = len(and_node.child_molecules)
            solved_reactants = sum(1 for or_node in and_node.child_molecules if or_node.is_solved)
            availability_score = solved_reactants / max(total_reactants, 1)
            
            # Get unsolved reactants
            unsolved_reactants = [
                or_node.smiles for or_node in and_node.child_molecules
                if not or_node.is_solved
            ]
            
            if not unsolved_reactants:
                # All reactants are solved
                return 1.0
            
            # Use Oracle reward to calculate chemical feasibility
            oracle_score = self.oracle.reward(
                self.inventory, unsolved_reactants, 
                self.visited_molecules, self.dead_molecules
            )
            print(f'unscaled oracle score: {oracle_score}')
            
            # Convert oracle score to positive score
            chemistry_score = max(0.0, 1.0 + oracle_score/14)
            depth_penalty_factor = 0.99 ** and_node.depth
            
            # Combine availability and chemistry scores
            # Weight: 40% chemistry + 60% availability
            base_score = 0.4 * chemistry_score + 0.6 * availability_score
            final_score = base_score * depth_penalty_factor
            and_node.feasibility_score = final_score
            
            print(f"Evaluation - Availability: {availability_score:.3f} ({solved_reactants}/{total_reactants}), "
                  f"Chemistry: {chemistry_score:.3f}, Depth: {and_node.depth} (penalty: {depth_penalty_factor:.3f}), "
                  f"Final: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            print(f"Chemical evaluation failed: {e}")
            return 0.4
    
    def _initialize_and_node_stats(self, and_node: ANDNode, score: float):
        """Initialize AND node statistics"""
        and_node.visit_count = 1
        and_node.total_value = score
        and_node.average_value = score
    
    def _backpropagate_and_node(self, and_node: ANDNode, reward: float, visited_nodes: set = None):
        """Backpropagate AND node statistics"""
        if visited_nodes is None:
            visited_nodes = set()
        
        # Prevent circular visits
        if and_node.reaction_id in visited_nodes:
            return
            
        visited_nodes.add(and_node.reaction_id)
        
        current_or = and_node.parent_molecule
        if current_or is None or current_or == self.root_or_node:
            return

        # Update all parent AND nodes
        for parent_and in current_or.parent_reactions:
            parent_and.visit_count += 1
            parent_and.total_value += reward
            parent_and.average_value = parent_and.total_value / parent_and.visit_count
            # Recursive propagation
            self._backpropagate_and_node(parent_and, reward, visited_nodes)
    
    def _update_global_solved_status(self):
        """Update global solution status"""
        changed = True
        
        while changed:
            changed = False
            newly_solved_this_round = set()
            
            for molecule_smiles, or_node in self.all_molecules.items():
                if or_node.is_solved:
                    continue
                
                for and_node in or_node.child_reactions:
                    if all(child_or.is_solved for child_or in and_node.child_molecules):
                        or_node.is_solved = True
                        or_node.solving_and_node = and_node
                        newly_solved_this_round.add(molecule_smiles)
                        changed = True
                        
                        mol_str = or_node.smiles[:30] + "..." if len(or_node.smiles) > 30 else or_node.smiles
                        print(f"✅ Solved: {mol_str}")
                        break
            
            if newly_solved_this_round:
                self._reevaluate_affected_and_nodes(newly_solved_this_round)
                self._cleanup_newly_solved_nodes(newly_solved_this_round)

    def _cleanup_newly_solved_nodes(self, newly_solved_molecules: set):
        """Clean up all child reactions of newly solved nodes"""
        cleaned_count = 0
        
        for molecule_smiles in newly_solved_molecules:
            or_node = self.all_molecules[molecule_smiles]
            
            # Clean up all child reactions of this OR node
            for and_node in list(or_node.child_reactions):
                # Remove from leaf_and_nodes if present
                if and_node in self.leaf_and_nodes:
                    self.leaf_and_nodes.discard(and_node)
                    and_node.is_leaf = False
                
                # Recursively clean up subtree
                self._cleanup_subtree(and_node)
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} AND nodes from {len(newly_solved_molecules)} newly solved molecules")

    def _reevaluate_affected_and_nodes(self, newly_solved_molecules):
        """Re-evaluate AND nodes affected by newly solved molecules"""
        for molecule_smiles in newly_solved_molecules:
            or_node = self.all_molecules[molecule_smiles]
            for parent_and in or_node.parent_reactions:
                # Re-evaluate this AND node
                old_score = parent_and.feasibility_score  
                new_score = self._evaluate_and_node_chemical(parent_and)
                parent_and.feasibility_score = new_score
                # Optional: update average_value for UCB calculation
                parent_and.average_value = new_score
                print(f"🔄 Updated score: {old_score:.3f} -> {new_score:.3f} for reaction")
    
    def _cleanup_subtree(self, solved_and_node):
        """Recursively clean up subtree of solved AND node"""
        for child_or in solved_and_node.child_molecules:
            # If child OR node has no unsolved parent reactions, clean its subtree
            unsolved_parents = [parent for parent in child_or.parent_reactions 
                              if not all(c.is_solved for c in parent.child_molecules)]
            
            if not unsolved_parents:  # No unsolved parent reactions
                for child_and in list(child_or.child_reactions):
                    if child_and in self.leaf_and_nodes:
                        self.leaf_and_nodes.discard(child_and)
                        child_and.is_leaf = False
                        # Recursively clean deeper subtrees
                        self._cleanup_subtree(child_and)
    
    def _extract_solution(self) -> Optional[Dict]:
        """Extract solution using recorded solving paths"""
        if not self.root_or_node.is_solved:
            print("No complete solution found")
            return None
        
        def build_solution_tree(or_node: ORNode, visited: set = None, max_depth: int = 30) -> Dict:
            # Initialize visited set
            if visited is None:
                visited = set()
            
            # Prevent infinite recursion
            if or_node.smiles in visited:
                return {
                    "type": "circular_reference",
                    "molecule": or_node.smiles,
                    "note": "Circular reference detected"
                }
            
            # Prevent excessive depth
            if max_depth <= 0:
                return {
                    "type": "max_depth_reached",
                    "molecule": or_node.smiles,
                    "note": "Maximum recursion depth reached"
                }
            
            # Distinguish building blocks from reaction-solved molecules
            if or_node.is_solved and not or_node.child_reactions:
                # True building block (inventory molecule)
                return {
                    "type": "building_block",
                    "molecule": or_node.smiles
                }
            
            # Use recorded solving path
            if or_node.solving_and_node is None:
                return {
                    "type": "no_solving_path",
                    "molecule": or_node.smiles,
                    "note": "Node marked as solved but no solving path recorded"
                }
            
            # Add current node to visited set (use copy to avoid affecting siblings)
            visited_copy = visited.copy()
            visited_copy.add(or_node.smiles)
            
            # Use recorded solving path
            solving_and_node = or_node.solving_and_node
            
            return {
                "type": "reaction",
                "molecule": or_node.smiles,
                "reaction_id": solving_and_node.reaction_id,
                "reactants": [build_solution_tree(child, visited_copy, max_depth-1) 
                            for child in solving_and_node.child_molecules],
                "feasibility_score": solving_and_node.feasibility_score,
                "visit_count": solving_and_node.visit_count
            }
        
        solution = build_solution_tree(self.root_or_node)
        print("✅ Extracted complete solution tree using recorded solving paths")
        return solution
    
    def _extract_partial_solution(self) -> Dict:
        """Extract partial solution for failure case analysis"""
        
        def build_partial_tree(or_node: ORNode, visited: set = None, max_depth: int = 30) -> Dict:
            if visited is None:
                visited = set()
            
            if or_node.smiles in visited or max_depth <= 0:
                return {"type": "circular_or_deep", "molecule": or_node.smiles}
            
            # Building block
            if or_node.is_solved and not or_node.child_reactions:
                return {"type": "building_block", "molecule": or_node.smiles}
            
            # Unsolved leaf node
            if not or_node.child_reactions:
                return {
                    "type": "unsolved_leaf", 
                    "molecule": or_node.smiles,
                    "note": "No synthetic routes attempted"
                }
            
            # Select best reaction path
            if or_node.is_solved and or_node.solving_and_node:
                best_and_node = or_node.solving_and_node
                status = "solved"
            else:
                # Select most promising AND node
                best_and_node = max(or_node.child_reactions, 
                                  key=lambda x: (x.feasibility_score, x.visit_count, -x.depth))
                status = "partial"
            
            visited_copy = visited.copy()
            visited_copy.add(or_node.smiles)
            
            return {
                "type": "reaction",
                "status": status,
                "molecule": or_node.smiles,
                "reaction_id": best_and_node.reaction_id,
                "reactants": [build_partial_tree(child, visited_copy, max_depth-1) 
                            for child in best_and_node.child_molecules],
                "feasibility_score": best_and_node.feasibility_score,
                "visit_count": best_and_node.visit_count,
                "depth": best_and_node.depth
            }
        
        # Collect statistics
        unsolved_leaves = []
        depth_stats = {"max_depth": 0, "depth_distribution": {}}
        
        for or_node in self.all_molecules.values():
            if not or_node.is_solved and not or_node.child_reactions:
                unsolved_leaves.append(or_node.smiles)
            
            for and_node in or_node.child_reactions:
                depth = and_node.depth
                depth_stats["max_depth"] = max(depth_stats["max_depth"], depth)
                depth_stats["depth_distribution"][depth] = depth_stats["depth_distribution"].get(depth, 0) + 1
        
        partial_tree = build_partial_tree(self.root_or_node)
        
        return {
            "type": "partial_solution",
            "target_molecule": self.root_or_node.smiles,
            "solution_tree": partial_tree,
            "statistics": {
                "total_and_nodes": self.total_and_nodes,
                "total_or_nodes": len(self.all_molecules),
                "unsolved_leaf_count": len(unsolved_leaves),
                "unsolved_leaves": unsolved_leaves[:10],
                "depth_stats": depth_stats,
                "leaf_and_nodes_remaining": len(self.leaf_and_nodes)
            },
            "analysis_hints": self._generate_failure_analysis_hints(unsolved_leaves, depth_stats)
        }

    def _generate_failure_analysis_hints(self, unsolved_leaves: List[str], depth_stats: Dict) -> List[str]:
        """Generate failure analysis hints"""
        hints = []
        
        if len(unsolved_leaves) > 10:
            hints.append(f"Too many unsolved leaves ({len(unsolved_leaves)}), may need broader search")
        
        if depth_stats["max_depth"] >= self.max_depth - 1:
            hints.append(f"Reached max depth ({self.max_depth}), may need deeper search")
        
        if len(depth_stats["depth_distribution"]) <= 2:
            hints.append("Search too narrow, may need more exploration")
        
        if self.total_and_nodes < 10:
            hints.append("Very few nodes explored, LLM may be generating poor routes")
        
        return hints

    def _extract_trivial_solution(self) -> Dict:
        """Extract trivial solution (target already purchasable)"""
        return {
            "type": "trivial",
            "molecule": self.root_or_node.smiles,
            "message": "Target molecule is already purchasable"
        }
    
    # Helper methods
    def _get_similar_routes(self, target_smiles: str, route_list: List, all_fps: List, 
                           num_examples: int = 3) -> List:
        """Get similar historical routes (RAG)"""
        try:
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
            similarity_metric = DataStructs.BulkTanimotoSimilarity
            
            target_fp = getfp(target_smiles)
            sims = similarity_metric(target_fp, all_fps)
            
            # Get most similar routes
            rag_tuples = list(zip(sims, route_list))
            rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:num_examples]
            
            return [t[1] for t in rag_tuples]
            
        except Exception as e:
            print(f"Failed to get similar routes: {e}")
            return route_list[:num_examples]
    
    def _is_and_node_fully_solved(self, and_node: ANDNode) -> bool:
        """Check if AND node is fully solved"""
        return all(child_or.is_solved for child_or in and_node.child_molecules)
    
    def _has_expandable_reactants(self, and_node: ANDNode) -> bool:
        """Check if AND node has expandable reactants"""
        return any(not or_node.is_solved for or_node in and_node.child_molecules)
    
    def _get_ucb_score(self, and_node: ANDNode, c_param: float = 0.5) -> float:
        """Calculate UCB1 score"""
        if and_node.visit_count == 0:
            return float('inf')
        
        if and_node.parent_molecule is None:
            return and_node.average_value
        
        parent_total_visits = sum(sibling.visit_count 
                                 for sibling in and_node.parent_molecule.child_reactions)
        
        exploitation = and_node.average_value
        exploration = c_param * math.sqrt(math.log(max(parent_total_visits, 1)) / 
                                         max(and_node.visit_count, 1))
        return exploitation + exploration
    
    def _select_expansion_target(self, unsolved_reactants: List[ORNode]) -> ORNode:
        """Select reactant most needing expansion"""
        # Prioritize molecules with no child reactions
        def priority_key(or_node):
            if not or_node.child_reactions:
                return (0, 0)  # Highest priority: no child reactions
            else:
                total_visits = sum(child.visit_count for child in or_node.child_reactions)
                return (1, total_visits)  # Secondary priority: by visit count
        
        return min(unsolved_reactants, key=priority_key)
    
    def _expand_or_node_with_llm(self, or_node: ORNode, route_list: List, all_fps: List) -> List[ANDNode]:
        """Generate initial AND nodes for root OR node"""
        generated_routes = self._generate_routes_with_llm(or_node, route_list, all_fps)
        return self._map_routes_to_tree(or_node, generated_routes, 0)

    def _find_matching_step(self, or_node: ORNode, steps: List[Dict]) -> Optional[Dict]:
        """Find matching route step for OR node"""
        target_smiles = or_node.smiles
        
        for step in steps:
            try:
                product = extract_molecules_from_output(step['Product'])[0]
                if sanitize_smiles(product) == target_smiles:
                    return step
            except:
                continue
        
        return None