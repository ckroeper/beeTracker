# Environment Setup Instructions

This guide will help you set up the Python environment for our machine learning application.
We are using Python **3.11.6** and several key libraries such as `ultralytics` (for YOLO), `streamlit` (for the frontend), `pandas`, and `opencv-python`. **We are not providing a Docker container**, since the client may be on macOS (Intel or Apple Silicon (Req. Pytorch MPS)), Windows, or Linux. tools and `pip`.

**Important:** PyTorch (with GPU acceleration) should be installed based on your machine’s configuration. Because the recommended installation steps differ for each OS and CUDA version, you need to follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install the correct build (with GPU support if applicable). Once you have PyTorch 2.3.0 and TorchVision 0.18.0 installed, you can proceed with the rest of the dependencies.

## Steps

1. **Install Python 3.11.6**

   Make sure Python 3.11.6 is installed on your system. We recommend using:

   - **Windows:** [Download from Python.org](https://www.python.org/downloads/) or use [Chocolatey](https://chocolatey.org/) with `choco install python`.
   - **macOS:** Use [Homebrew](https://brew.sh/) with `brew install python@3.11` or [pyenv](https://github.com/pyenv/pyenv). On Apple Silicon, you must ensure you have the correct ARM-compatible Python.
   - **Linux:** Use your distribution’s package manager.

   **Note**: You may also use [anaconda](https://docs.anaconda.com/miniconda/install/) to setup and install your Python environment. We would recommend this, but for simplicity's sake, we provide pip instructions.

2. **Create a Virtual Environment**

   Create and activate a virtual environment to isolate project dependencies.

   ```bash
   # Ensure python3.11 points to the Python 3.11.6 installation or adjust path accordingly.
   python3.11 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install PyTorch and TorchVision**

   We recommend going to the [PyTorch website](https://pytorch.org/get-started/locally/) and following their instructions. For example, if you are on Linux with a CUDA-capable GPU:

   ```bash
   # Example (Linux + CUDA): Adjust based on instructions from pytorch.org
   pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

   If you’re on an Apple M-series machine, you might use:

   ```bash
   # Example for Apple Silicon (M1/M2):
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   If you’re on CPU-only, you can just install the CPU wheels:

   ```bash
   pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

   **Check the PyTorch website for the exact command that suits your system and CUDA version.**

4. **Install Other Dependencies**

   Once PyTorch is installed, install all other dependencies from `requirements.txt`:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This will install:

   - ultralytics==8.3.17 (for YOLO models)
   - streamlit==1.40.2 (for web interface)
   - pandas
   - opencv-python

5. **Verify the Installation**

   Check versions:

   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import ultralytics; print(ultralytics.__version__)"
   python -c "import streamlit; print(streamlit.__version__)"
   python -c "import pandas; print(pandas.__version__)"
   python -c "import cv2; print(cv2.__version__)"
   ```

   Make sure they match the expected versions (or at least that there are no errors).

6. **Run Your Application**

   Run the provided code with:

   ```bash
   streamlit run main.py
   ```

   Ensure `main.py` is the entry point of your application if you modified the code.

## Additional Notes

1. **If you encounter performance issues or need GPU acceleration, ensure you have the correct PyTorch build.**
2. **Another optimization would be to export the model from PyTorch format (`.pt`) and instead use one of:**

- Onnx (Cross-platform)
- TensorRT (CUDA)
- CoreML (MPS)
- etc. [other supported formats](https://docs.ultralytics.com/modes/export/#arguments)

3. **We did not export the model due to deployment concerns:**

- Export model to your selected format (that works with your compute environment).
- Modify `main.py`: Simply point the line that says `model = YOLO(PUT_PATH_HERE)` in the load_model function.
- Rerun the application.

4. **Anaconda distributions for ARM may additionally help resolve dependency issues on apple silicon chips.**
