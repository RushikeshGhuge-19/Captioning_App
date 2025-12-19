import gradio as gr
import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
from torchvision import transforms, models
from PIL import Image
from torch import nn
from huggingface_hub import hf_hub_download

nltk.download('punkt', quiet=True)

# ---------- Vocabulary ----------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return word_tokenize(text.lower())

    def numericalize(self, text):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in self.tokenizer(text)]

# ---------- Attention ----------
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((features, hidden), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

# ---------- Captioning Model ----------
class AttentionCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.fc_features = nn.Linear(2048, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(embed_size, hidden_size)
        self.lstm = nn.LSTMCell(embed_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        pass  # Not needed in inference

    def generate_caption(self, feature, vocab, max_len=20):
        result = []
        input_word = torch.tensor([vocab.stoi["<SOS>"]]).to(device)
        feature = self.fc_features(feature)
    
        h = torch.zeros(1, 512).to(device)
        c = torch.zeros(1, 512).to(device)
    
        for _ in range(max_len):
            embedded = self.embed(input_word)  # [1, embed_size]
            attn_weights = self.attention(feature.unsqueeze(1), h)
            context = (attn_weights.unsqueeze(2) * feature.unsqueeze(1)).sum(1)  # [1, embed_size]
    
            lstm_input = torch.cat((embedded, context), dim=1)  # Both are [1, embed_size]
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc_out(h)
            predicted = output.argmax(1)
            word = vocab.itos[predicted.item()]
            if word == "<EOS>":
                break
            result.append(word)
            input_word = predicted
    
        return " ".join(result)


# ---------- Load Model and Vocab ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = hf_hub_download(repo_id="aditii17/Captioning", filename="caption_model.pth")
vocab_path = hf_hub_download(repo_id="aditii17/Captioning", filename="vocab.pkl")

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

model = AttentionCaptionModel(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- Feature Extractor ----------
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

# ---------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Inference Function ----------
def predict_caption(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(image).squeeze()
        caption = model.generate_caption(feature.unsqueeze(0), vocab)
    return caption

# ---------- Gradio Interface ----------
gr.Interface(fn=predict_caption, inputs=gr.Image(type="pil"), outputs="text", title="🖼️ Image Captioning").launch()
