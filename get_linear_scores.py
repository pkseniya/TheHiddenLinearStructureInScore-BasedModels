import argparse
import torch

from os.path import join

from linear_structure.utils import get_gaussian_score
from generate import StackedRandomGenerator

def configure_arg_parser(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("scores_dir", type=str, help="Dir to save scores") 
    parser.add_argument("parameters_dir", type=str, help="Calculated score parameters directory") 
    return parser 

def main(scores_dir, parameters_dir):
    device = "cuda"
    batch_seeds = [0, 1, 2, 3, 4]
    sigmas = [80.0, 57.6, 40.8, 28.4, 19.4, 12.9, 8.4, 5.3, 3.3, 1.1, 0.3]
    steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    
    rnd = StackedRandomGenerator(device, batch_seeds)
    latents = rnd.randn([5, 3, 32, 32], device=device)

    mu = torch.load(join(parameters_dir, "mu.pt"), map_location=device)
    U = torch.load(join(parameters_dir, "U.pt"), map_location=device)
    lambdas = torch.load(join(parameters_dir, "lambdas.pt"), map_location=device)

    for step, sigma in zip(steps, sigmas):
        score = get_gaussian_score(latents, sigma, mu, U, lambdas)

        torch.save(score.cpu(), join(scores_dir, f"{step}.pt"))


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(**vars(_args))

