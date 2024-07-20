import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stHeader {
        background-color: #262730;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stImage {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #262730;
        color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
@st.cache_resource
def train_model():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # Train for 5 epochs
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

# Function to load or train the model
@st.cache_resource
def get_model():
    try:
        model = SimpleCNN()
        model.load_state_dict(torch.load('cifar10_model.pth'))
        model.eval()
    except:
        model = train_model()
        torch.save(model.state_dict(), 'cifar10_model.pth')
    return model

# Sidebar
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main content
st.markdown("<h1 class='stHeader'>CIFAR-10 Image Classification</h1>", unsafe_allow_html=True)

# Drag and drop section
col1, col2, col3 = st.columns([1,2,1])
# with col2:
#     st.markdown("<div class='upload-box'>Drag and drop image here</div>", unsafe_allow_html=True)

# Display uploaded image and make prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<div class='stImage'>", unsafe_allow_html=True)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Load model and make prediction
    model = get_model()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Display results in sidebar
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    _, predicted = torch.max(output, 1)
    st.sidebar.markdown("<div class='stSuccess'>", unsafe_allow_html=True)
    st.sidebar.write(f"Best Prediction: {classes[predicted.item()]}")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    st.sidebar.write("Prediction Probabilities:")
    for i, prob in enumerate(probabilities):
        st.sidebar.write(f"{classes[i]}: {prob.item():.2%}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Created with Streamlit and PyTorch</p>", unsafe_allow_html=True)