# AFM Tomography Visualization Tool

A comprehensive tool for visualizing and analyzing Atomic Force Microscopy (AFM) depth profiles. This application provides interactive visualization capabilities for AFM tomography data, enabling researchers to explore material properties at different depths.

## Features

- **Data Loading**: Load AFM depth profile images from different directories
- **Data Processing**: Apply interpolation and contrast enhancement to improve visualization
- **Multiple Visualization Modes**:
  - Profile View: Examine individual depth profiles
  - Heatmap View: Visualize all depth profiles in a 2D heatmap
  - 3D Surface: Explore the full 3D structure of the material
- **Interactive Controls**: Adjust visualization parameters in real-time
- **Scientific Analysis**: Tools designed for detailed material property analysis

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/afm-tomography.git
cd afm-tomography

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/afm-tomography.git
cd afm-tomography

# Create and activate conda environment
conda env create -f environment.yml
conda activate afm_tomo_env
```

## Usage

```bash
python app.py
```

Then open your web browser and navigate to http://127.0.0.1:8050/

## Project Structure

```
├── app.py                 # Main Dash application
├── data_loader.py         # Functions for loading AFM depth profile data
├── data_processor.py      # Data processing and interpolation functions
├── colormaps.py           # Custom colormaps for AFM visualization
├── visualizations.py      # Visualization components
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment specification
├── data/                  # Sample AFM depth profile images
│   └── *.jpg              # AFM images at different depths
└── README.md              # Project documentation
```

## Data Format

The application expects AFM depth profile images with depth information in the filename (e.g., `surface-10micron-deep.jpg`). The images should contain AFM scans at different depths, and the application will extract the central row of each image as the 1D profile.

## Citation

If you use this tool in your research, please cite:

```
Author, A. (2025). AFM Tomography Visualization Tool: A comprehensive platform for analyzing material properties at different depths. Journal of Scientific Visualization, XX(X), XXX-XXX.
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
