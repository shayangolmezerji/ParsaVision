# ParsaVision: The Universal AI Server 🧠

ParsaVision is a production-ready template for building and serving AI models. It demonstrates a professional, decoupled machine learning pipeline where model training is separated from the serving infrastructure.

Built with a focus on best practices, this project showcases a clean architecture using Python, FastAPI, and PyTorch.

### ✨ Features

  * **Decoupled Architecture**: Separates the training (`trainer.py`) and serving (`ai_server.py`) pipelines.
  * **Robust CLI**: Uses `click` for a powerful and user-friendly command-line interface.
  * **Professional Logging**: Implements Python's `logging` module for clear, timestamped server output.
  * **Health Check Endpoint**: Includes a `/health` endpoint for monitoring server status.
  * **Pre-trained Model**: Trains a simple CNN on the MNIST dataset, making it a "smart" AI right out of the box.

### 🚀 Getting Started

#### 1\. Clone the repository

```bash
git clone https://github.com/shayangolmezerji/ParsaVision.git
cd ParsaVision
```

#### 2\. Install Dependencies

This project requires a few key libraries. It is highly recommended to use a virtual environment.

```bash
python3 -m venv env
source env/bin/activate
pip install torch torchvision numpy click uvicorn[standard] fastapi
```

#### 3\. Train the Model

This step downloads the MNIST dataset and trains a CNN model, saving the trained model to a file named `simple_cnn.pt`.

```bash
python3 trainer.py
```

#### 4\. Run the Server

Start the AI server, pointing it to the trained model file.

```bash
python3 ai_server.py --model-path simple_cnn.pt
```

### 🧠 How to Test

With the server running, you can test it by sending a pre-processed image to the `/predict` endpoint.

1.  **Save a test image:** Download an image of a handwritten digit (e.g., an MNIST image) and save it as `test.png` on your desktop.
2.  **Use the preprocessor:** The `preprocess.py` script will convert the image into a format the API understands (a list of 784 numbers).
3.  **Send the request:** Use `curl` to pipe the output of the preprocessor to the running API server.

<!-- end list -->

```bash
python3 preprocess.py ~/Desktop/test.png | curl -X POST "http://0.0.0.0:8000/predict" -H "Content-Type: application/json" -d @-
```

The server will respond with the predicted digit.

### 📜 License

This project is licensed under the [DON'T BE A DICK PUBLIC LICENSE](LICENSE.md).

### 👨‍💻 Author

Made with ❤️ by [Shayan Golmezerji](https://github.com/shayangolmezerji)
