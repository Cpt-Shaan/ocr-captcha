import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import soundfile as sf
import io
from IPython.display import Audio
from kokoro import KPipeline
import gdown

MODEL_PATH = "ocr_captv1.pth"
DRIVE_FILE_ID = "1ivVAxuZw3J1Spcc806JgzXNJm44nhvQQ"  # ← replace with your actual file ID

# If model is not already present, download from Google Drive
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ---------------------------
# 1. Load model definition
# ---------------------------

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 2, stride=1, padding=0),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(512 * 2, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, w, c * h)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# ---------------------------
# 2. Greedy Decoder
# ---------------------------
def greedy_decoder(output, idx_to_char, blank_idx):
    argmaxes = torch.argmax(output, dim=2)
    decodes = []
    for i in range(argmaxes.size(0)):
        decode = []
        prev = -1
        for j in range(argmaxes.size(1)):
            curr = argmaxes[i, j].item()
            if curr != prev and curr != blank_idx:
                decode.append(idx_to_char[curr])
            prev = curr
        decodes.append("".join(decode))
    return decodes


# ---------------------------
# 3. Load OCR Model + TTS
# ---------------------------
@st.cache_resource
def load_model_and_utils():
    # Define your characters same as training
    characters = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
    idx_to_char = {i: c for i, c in enumerate(characters)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(len(characters) + 1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tts_pipeline = KPipeline(lang_code='a')  # English voice model
    st.success("OCR Model & TTS Model loaded successfully ✅")
    return model, idx_to_char, characters, device, tts_pipeline


# ---------------------------
# 4. OCR function
# ---------------------------
def ocr_infer(model, img, device, idx_to_char, characters):
    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img).log_softmax(2)
    text = greedy_decoder(output.cpu(), idx_to_char, blank_idx=len(characters))[0]
    return text


# ---------------------------
# 5. Text-to-Speech via Kokoro
# ---------------------------
def speak_text(pipeline, text):
    generator = pipeline(f"The captcha is {text}", voice='af_heart')
    audio_combined = []
    for _, _, audio in generator:
        audio_combined.extend(audio)
    audio_data = torch.tensor(audio_combined).numpy()
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, audio_data, 24000, format='WAV')
    wav_bytes.seek(0)
    return wav_bytes


# ---------------------------
# 6. Streamlit App
# ---------------------------
st.set_page_config(page_title="Captcha OCR + TTS", page_icon="🔠", layout="centered")
st.title("🧠 Captcha OCR + Speech (Kokoro TTS)")

st.write("Upload a captcha image below to recognize and **hear** it.")

model, idx_to_char, characters, device, tts_pipeline = load_model_and_utils()

uploaded = st.file_uploader("Upload Captcha Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    st.image(img, caption="Uploaded Captcha", use_column_width=True)

    with st.spinner("Recognizing Captcha..."):
        text = ocr_infer(model, img, device, idx_to_char, characters)

    st.success(f"**Recognized Text:** `{text}`")

    with st.spinner("Generating Speech..."):
        audio_bytes = speak_text(tts_pipeline, text)

    st.audio(audio_bytes, format="audio/wav", start_time=0)
