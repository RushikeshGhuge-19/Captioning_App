
---

## 🧠 Image Captioning with Attention and ResNet 

This project is about teaching a computer how to look at a picture and describe it in words, like how a human would. It uses deep learning and works in two steps: first, it understands the picture using a model called **ResNet**, and then it writes a caption using another model called **LSTM with Attention**.

---

### ✨ What This Project Does

* Looks at an image
* Understands what’s in the image
* Writes a short sentence (caption) about the image

---

### 🧰 Tools and Technologies Used

* **Python** – programming language
* **PyTorch** – deep learning library
* **ResNet50** – used to extract image features
* **LSTM (with Attention)** – used to generate the captions
* **NLTK** – helps with processing words

---

### 🧠 How It Works

1. **ResNet50** looks at the image and turns it into numbers (features).
2. **LSTM with Attention** takes those numbers and creates a sentence.
3. The model learns by looking at thousands of image-caption pairs.
4. After learning, it can generate captions for new images.

---

### 📁 Code Overview

* `Vocabulary`: Helps the model understand and convert words into numbers.
* `Dataset`: Loads images and their captions, and prepares them for training.
* `EncoderCNN`: Uses ResNet50 to process the images.
* `DecoderRNN`: Uses LSTM and attention to generate captions.
* `train.py`: Main file that runs the training.

---

### 🧪 How to Run

1. Install the libraries:

   ```bash
   pip install torch torchvision nltk Pillow
   ```
2. Put your images and captions in the dataset folder.
3. Run `train.py` to train the model.
4. Use the model to generate captions for new images.

---

### 📸 Output Example

Input: 🖼️ (Picture of a dog playing in a park)

Output: `"A dog is running on the grass"`
---
### Screenshots

<img width="1878" height="1041" alt="22 07 2025_17 40 12_REC" src="https://github.com/user-attachments/assets/6eb824b6-b71a-4342-a212-74597d0624e3" />
