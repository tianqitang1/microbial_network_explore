import numpy as np
import os
import yaml
from yaml import CLoader
import shutil

# import src.utils.utils as utils
import utils.simulation as simulation

if __name__ == "__main__":
    # Read the config file and set the parameters
    config = yaml.load(open("scripts/config.yaml", "r"), Loader=CLoader)

    n_vertices = config["network"]["n_vertices"]
    avg_degree = config["network"]["avg_degree"]
    network_type = config["network"]["network_type"]
    interaction_type = config["network"]["interaction_type"]
    max_interaction_strength = config["network"]["max_interaction_strength"]

    time_points = config["simulation"]["time_points"]
    time_step = config["simulation"]["time_step"]
    downsample = config["simulation"]["downsample"]
    noise_var = config["simulation"]["noise_var"]
    scale_simulation = config["simulation"]["scale_simulation"]

    path = config["path"]["data_dir"]
    saving_dir_path = os.path.join(path, f"n{n_vertices}_k{avg_degree}_{network_type}_{interaction_type}_{max_interaction_strength}_{time_points}")
    if not os.path.exists(saving_dir_path):
        os.makedirs(saving_dir_path)
    shutil.copyfile("scripts/config.yaml", os.path.join(saving_dir_path, "config.yaml"))  # Copy the config file to the data directory

    # Simulate data using the gLV model
    z, x, y, adj, M = simulation.simulate_glv(
        num_taxa=n_vertices,
        avg_degree=avg_degree,
        time_points=time_points,
        time_step=time_step,
        downsample=downsample,
        noise_var=noise_var,
        network_type=network_type,
        interaction_type=interaction_type,
        max_interaction_strength=max_interaction_strength,
    )

    # Scale the simulated observed read counts
    y = y * scale_simulation

    # Save data using numpy
    np.save(os.path.join(saving_dir_path, "z.npy"), z)
    np.save(os.path.join(saving_dir_path, "x.npy"), x)
    np.save(os.path.join(saving_dir_path, "y.npy"), y)
    np.save(os.path.join(saving_dir_path, "adj.npy"), adj)
    np.save(os.path.join(saving_dir_path, "M.npy"), M)
