from rl_david_silver_2015.mdp3.sampler_generator import sample_mdp_batched_generator
from rl_david_silver_2015.mdp3.tabular_mdpf import SAMPLE_TABULAR_STARTING_STATE, SAMPLE_TABULAR_MDP, SAMPLE_TABULAR_POLICY, TabularMDPFramework


for x in sample_mdp_batched_generator(SAMPLE_TABULAR_MDP, SAMPLE_TABULAR_POLICY, TabularMDPFramework, SAMPLE_TABULAR_STARTING_STATE):
    print(x)