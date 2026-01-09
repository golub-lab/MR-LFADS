<h1 align="center">
  Multi-Regional Latent Factors via Dynamical Systems Analysis
</h1>

  Multi-Regional Latent Factors via Dynamical Systems Analysis (MR-LFADS) is a data-driven dynamical systems model for disentangling and accurately identifying inter-regional communication in multi-region neural recordings. MR-LFADS is a sequential variational autoencoder with region-specific recurrent networks [1], structured information bottlenecks, and rate-based communication, enabling it to separate communication signals from local dynamics and inputs from unobserved regions. It outperforms existing approaches on simulated data and predicts brain-wide effects in large-scale electrophysiology, positioning it as a powerful tool for discovering principles of brain-wide information processing. 

<p align="center">
  <a href="https://icml.cc/virtual/2025/poster/45466">
    <img src="https://img.shields.io/badge/Publication-View%20Paper-blue?style=for-the-badge" alt="Publication">
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#installation-and-setup">Installation and Setup</a></li>
    <li><a href="#tutorials">Tutorials</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#reference">Reference</a></li>
  </ol>
</details>


## Installation and Setup

1. Install required packages via
   ```sh
   python -m pip install -r requirements.txt
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/golub-lab/MR-LFADS.git
   ```
3. Setup the folder paths.


<!-- TUTORIALS -->
## Tutorials

Put tutorials here.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

## Reference
[1] Chethan Pandarinath, Daniel J O’Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D Stavisky, Jonathan C Kao, Eric M Trautmann, Matthew T Kaufman, Stephen I Ryu, Leigh R Hochberg, et al. Inferring single-trial neural population dynamics using sequential auto-encoders. Nature Methods, 15(10):805–815, 2018.
