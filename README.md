<h2 align="center">
  Multi-Region Latent Factors via Dynamical Systems Analysis
</h2>

  Neural recording technologies now enable simultaneous recording of population activity across many brain regions, motivating the development of data-driven models of communication between brain regions. However, existing models can struggle to disentangle the sources that influence recorded neural populations, leading to inaccurate portraits of inter-regional communication. Here, we introduce Multi-Region Latent Factor Analysis via Dynamical Systems (MR-LFADS), a sequential variational autoencoder designed to disentangle inter-regional communication, inputs from unobserved regions, and local neural population dynamics. In our paper [1], we showed that MR-LFADS outperforms existing approaches at identifying communication across dozens of simulations of task-trained multi-region networks. We also showed that, when applied to large-scale electrophysiology, MR-LFADS predicts brain-wide effects of circuit perturbations that were held out during model fitting. These validations on synthetic and real neural data position MR-LFADS as a promising tool for discovering principles of brain-wide information processing.

<p align="center">
  <a href="https://arxiv.org/abs/2506.19094">
    <img src="https://img.shields.io/badge/Publication-Our%20ICML%202025%20paper-blue?style=for-the-badge" alt="Publication">
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#installation-and-setup">Installation and Setup</a></li>
    <li><a href="#tutorial">Tutorial</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>


## Installation and Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/golub-lab/MR-LFADS.git
   cd MR-LFADS/
   ```
2. Create and activate a conda environment:
   ```sh
   conda create -n mrlfads_env python=3.8.5 -y
   conda activate mrlfads_env
   ```
3. Install system dependencies required for `torch.compile` (Linux only):

  > **Note:** The following commands are specific to **Ubuntu/Debian** systems (`apt-get`).  
  > If you are on another distribution (e.g., RHEL/CentOS/Amazon Linux using `yum`/`dnf`), or if `gcc` and `g++` are already installed, you can **skip this step**.

  Ubuntu / Debian:
   ```sh
   sudo apt-get update
   sudo apt-get install -y build-essential
   ```

  Verify required tools:
  ```sh
  gcc --version
  g++ --version
  make --version
  ```

  Set compiler environment variables:
  ```sh
   export CC=gcc
   export CXX=g++
   ```

4. Create `config/paths.py` and define the following paths to match your local environment:
   ```sh
   homepath = "/absolute/path/to/MR-LFADS"   # root directory of the repository
   datapath = "/absolute/path/to/data"       # directory for datasets
   resultpath = "/absolute/path/to/results"  # directory for training outputs
   ```

5. Install required packages for the `mrlfads` module:
   ```sh
   python -m pip install -r requirements.txt
   pip install -e .
   ```

<!-- TUTORIAL -->
## Tutorial

See `tutorials/` for step-by-step instructions on how to prepare your dataset, configurate the MR-LFADS model, and perform inference on the trained model.

<!-- LICENSE -->
## License

See LICENCE file.

<!-- CONTACT -->
## Contact

For questions or feedback, please contact: belleliu@uw.edu

## Reference
[1] Belle Liu, Jacob Sacks, and Matthew D. Golub. "Multi-Region Latent Factors via Dynamical Systems Analysis." ICML 2025. https://arxiv.org/abs/2506.19094
