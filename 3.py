import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (same as in your training code)
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)

# Load the best model weights
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

# Define the transform (same as in your training code)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict the class of an image
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    with torch.no_grad():
        output = model(image)
        prediction = output.item()  # Get the predicted probability
    return prediction

# Streamlit UI
st.title("Interactive Image Classification with ResNet50")
st.write("Upload an image to classify and see the predicted class.")

# Upload image for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the class of the image
    prediction = predict_image(image)
    
    # Display the result as binary classification with bold text and color
    if prediction > 0.5:
        prediction_class = "Antibiotic required"
        st.markdown(f"### **Prediction: {prediction_class}**")
    else:
        prediction_class = "Antibiotic not required"
        st.markdown(f"### **Prediction: {prediction_class}**")

# Add a button to reset or clear the image upload
if st.button("Clear Image"):
    st.experimental_rerun()
