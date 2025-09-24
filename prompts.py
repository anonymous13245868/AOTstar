"""
LLM Prompt Templates for Retrosynthesis Planning
"""

def get_initial_prompt():
    """Get the initial system prompt for retrosynthesis"""
    return '''
        You are a professional chemist specializing in synthesis analysis. Your task is to propose a retrosynthesis route for a target molecule provided in SMILES format.

        Definition:
        A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

        Key concepts:
        - Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
        - Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
        - Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
        - Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

        Process:
        1. Initialization: Start with the molecule set = [target molecule].
        2. Iteration:
            - Select one non-purchasable molecule from the molecule set (the product).
            - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
            - Remove the product molecule from the set.
            - Add the reactants to the set.
        3. Termination: Continue until all molecules in the set are commercially purchasable.
        '''


def get_task_description(target_smiles: str, examples: str, expansion_routes: int):
    """Format the task description with target and examples"""
    return f'''
        My target molecule is: {target_smiles}

        To assist you with the format, example retrosynthesis routes are provided:
        {examples}

        Please propose {expansion_routes} different retrosynthesis routes for my target molecule. The provided reference routes may be helpful.
        '''


def get_requirements():
    """Get the formatting requirements for the response"""
    return '''
        You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
        At the first step, the molecule set should be the target molecules set given by the user. Here is an example:

        <PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
        <EXPLANATION>: Explain the plan. </EXPLANATION>
        <ROUTE>
        [   
            {
                'Molecule set': "[Target Molecule]",
                'Rational': Step analysis,
                'Product': "[Product molecule]",
                'Reaction': "[Reaction template]",
                'Reactants': "[Reactant1, Reactant2]",
                'Updated molecule set': "[Reactant1, Reactant2]"
            },
            {
                'Molecule set': "[Reactant1, Reactant2]",
                'Rational': Step analysis,
                'Product': "[Product molecule]",
                'Reaction': "[Reaction template]",
                'Reactants': "[subReactant1, subReactant2]",
                'Updated molecule set': "[Reactant1, subReactant1, subReactant2]"
            }
        ]
        </ROUTE>

        \n\n
        Requirements: 1. The 'Molecule set' contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the 'Updated molecule set' from the previous step.\n
        2. The 'Rational' part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with \'\'\n
        3. 'Product' is the molecule we plan to synthesize in this step. It should be from the 'Molecule set'. The molecule should be a molecule from the 'Molecule set' in a list. The molecule smiles should be wrapped with \'\'.\n
        4. 'Reaction' is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ['Product>>Reactant1.Reactant2'].\n
        5. 'Reactants' are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with \'\'.\n
        6. The 'Updated molecule set' should be molecules we need to purchase or synthesize after taking this reaction. To get the 'Updated molecule set', you need to remove the product molecule from the 'Molecule set' and then add the reactants in this step into it. In the last step, all the molecules in the 'Updated molecule set' should be purchasable.\n
        7. In the <PLAN>, you should analyze the target molecule and plan for the whole route.\n
        8. In the <EXPLANATION>, you should analyze the plan.\n'''


def construct_full_prompt(target_smiles: str, examples: str, expansion_routes: int):
    """Combine all prompt parts into complete prompt"""
    initial_prompt = get_initial_prompt()
    task_description = get_task_description(target_smiles, examples, expansion_routes)
    requirements = get_requirements()
    
    return initial_prompt + task_description + requirements