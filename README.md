Ayna ML Internship Assignment: Conditional UNet for Polygon Colorization
This project implements a Conditional UNet model trained to colorize polygon images based on a given color name. It was built as part of the Ayna ML Internship take-home assignment.

📌 Problem Statement
Given an input polygon image (e.g., triangle, square, octagon) and a color name (e.g., red, green, blue), the model should output the same polygon filled with the specified color.

🏗️ Model Architecture
Implemented a custom UNet in PyTorch.

Conditioning on color is done by:

Embedding the color name (e.g., "red", "blue") into a vector.

Expanding and concatenating the embedding with the input image at the first layer.

Output is a colorized version of the input polygon image.

🧪 Dataset
Dataset includes:

inputs/: polygon outlines.

outputs/: ground truth images filled with correct colors.

data.json: mapping between input image, color, and expected output.

Split into:

training/

validation/

⚙️ Training Details
Parameter	Value
Optimizer	Adam
Learning Rate	1e-3
Loss Function	MSELoss
Epochs	20
Batch Size	16
Image Size	128x128
Color Embedding Dim	16

Training was tracked using Weights & Biases.

Loss steadily decreased and model produced increasingly accurate outputs.

📊 Results
Model was able to colorize polygons accurately in most cases.

Trained on Google Colab with GPU (T4).

Observed failure cases when:

Color shades are very similar (e.g., pink vs red).

Polygon edges are fuzzy or incomplete.

Input Polygon	Color	Output (Generated)
"green"	

📁 Project Structure
Copy
Edit
├── dataset/
├── model/
│   └── unet.py
├── train.py
├── utils.py
├── inference.ipynb
├── colored_polygon_model.pth
└── README.md
🧠 Key Learnings
How to build a UNet from scratch.

Techniques for conditioning vision models with non-visual input.

Dataset handling, augmentation, and mapping with JSON.

Integration with wandb for effective experiment tracking.

