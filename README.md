# Finding Waldo in Foundation Models

## Getting started
The code runs Python 3.10+. Here are some installation instructions:
Install [miniforge](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Follow the instruction online, most likely you will execute the following commands:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash ~/Miniforge3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Install Python in a new conda environment (be mindful to install a Python version compatible with Pytorch):
```bash
conda create -n myenv python==3.10
conda activate myenv
```
Install the repository (the ``pretrained" dependencies are optional but allow for a faster download of weights):
```bash
git clone <repo url>
cd <repo path>
pip install -e ".[pretrained]"
```
To install the development and visualization dependencies, you can swap the previous command for the following one:
```bash
pip install -e ".[pretrained,dev,visual]"
```

#### Accelerate specific instructions
The accelerate package can be used to download and distribute models from the HuggingFace Transformers library. After installing it, one needs to configure it. Follow the instruction online [configure-accelerate](https://huggingface.co/docs/accelerate/en/basic_tutorials/install), most likely you will execute the following command and answer the questions prompted to you:
```bash
accelerate config
```

### Using HuggingFace pretrained models
Some models are gated, e.g., the Llama and Gemma ones. Users should request the access and login to the huggingface hub to use them in the scripts. See https://huggingface.co/docs/hub/en/models-gated for more information.

## Acknowledgements

## Contact
If you have any questions, feel free to reach out at [```ambroiseodonnattechnologie@gmail.com```](mailto:ambroiseodonnattechnologie@gmail.com).

## Citation

```