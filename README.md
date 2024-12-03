# Waveform Generator Project

This project aims to generate gravitational waveforms using Generative Adversarial Networks (GANs), alongside other machine learning models, to help simulate and analyze gravitational wave data. The repository includes modules for waveform data generation, loading, GAN training, length estimation, and more.

## Features

- **Waveform Generation**: Generate gravitational waveforms given physical parameters such as mass ratio and spin.
- **Custom GAN Model**: Train a GAN to generate waveforms conditioned on specific physical parameters.
- **Length Estimator**: Predict the appropriate length for generated waveforms using machine learning.
- **Custom CNN model**: Train a CNN to generate waveforms conditioned on specific physical parameters.
- **Custom MLP model**: Train a MLP to generate waveforms conditioned on specific physical parameters.

## Installation

### Prerequisites
- **Python 3.8+**
- **Miniconda (Recommended)** for environment management.

### Setting Up the Environment
To set up the required environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/diego-dom95/GAN.git
   cd waveform_generator_project
   ```

2. Install PyTorch with CUDA 12.1 support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install dependencies: If using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
If using conda:
   ```bash
   conda create --name waveform_env python=3.8.0
   conda activate waveform_env
   pip install -r requirements.txt
   ```

## Usage
The project comes with a .ipynb file that shows how to generate data, train the model and run inference.

### Data Preparation
All the generated waveform data files (in HDF5 format) are automatically placed in the waveform_data/ directory.

## Dependencies
All dependencies are listed in the requirements.txt file. The main libraries used include:

- **PyTorch (with CUDA support)
- **NumPy
- **Matplotlib (for plotting)
- **tqdm (for loading bars)
- **h5py (for handling HDF5 files)
- **PyCBC (for gravitational waveform modeling)
- **SciPy (for signal resampling)

## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). See the LICENSE file for more details.

## Contributing
Contributions are welcome! If you find any bugs or would like to add new features, feel free to open an issue or submit a pull request.

## Contact
For questions or discussions, feel free to reach out to:

- **Diego Dominguez: [diego@gw.phys.sci.isct.ac.jp]
- **GitHub: [https://github.com/diego-dom95](https://github.com/diego-dom95)
