import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os
import smtplib
import matplotlib.pyplot as plt  # Import Matplotlib
from io import BytesIO
import base64

app = Flask(__name__)

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image):
    # Resize the image to (224, 224) as required by the model
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalize the image
    return image

def generate_graphs(confidence_score):
    # Scatter Plot
    x_values = np.arange(1, 11)
    y_values = np.random.randint(1, 100, size=10) * confidence_score / 100  # Adjust data based on confidence score
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values)
    plt.xlabel('Index')
    plt.ylabel('Data Value')
    scatter_plot_img = get_image_base64(plt)

    # Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(x_values, y_values)
    plt.xlabel('Index')
    plt.ylabel('Data Value')
    bar_chart_img = get_image_base64(plt)

    # Pie Chart
    labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']
    sizes = [10, 20, 30, 20, 20]  
    if confidence_score >= 70:
        sizes = [62, 30, 20, 70, 90]  
    elif confidence_score >= 50:
        sizes = [15, 25, 20, 25, 15]
    elif confidence_score >= 30:
        sizes = [10, 20, 30, 25, 15]
    elif confidence_score >= 10:
        sizes = [5, 15, 25, 30, 25]
    else:
        sizes = [0, 10, 20, 30, 40]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    pie_chart_img = get_image_base64(plt)

    return scatter_plot_img, bar_chart_img, pie_chart_img

def get_image_base64(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

def send_email(fromaddr, toaddrs, subject, message):
    msg = f"From: {fromaddr}\r\nTo: {', '.join(toaddrs)}\r\nSubject: {subject}\r\n\r\n{message}"
    server = smtplib.SMTP('localhost', 1025)  # Change this to your SMTP server details
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded')
        
        file = request.files['file']
        file_type = request.form.get('file_type')

        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file_type == 'image':
            # Image processing logic
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = preprocess_image(image)
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index] * 100
            
            # Generate graphs based on confidence score
            scatter_plot_img, bar_chart_img, pie_chart_img = generate_graphs(confidence_score)

            # Example email sending
            subject = f"Prediction Result: {class_name[2:]}"
            message = f"The model predicted {class_name[2:]} with {confidence_score:.2f}% confidence."
            send_email('sender@example.com', ['poloce@example.com'], subject, message)
            
            return render_template('index.html', class_name=class_name[2:], confidence_score=f"{confidence_score:.2f}%",
                                   scatter_plot_img=scatter_plot_img, bar_chart_img=bar_chart_img, pie_chart_img=pie_chart_img)
        
        elif file_type == 'video':
            # Save the video file to disk
            video_path = "uploaded_video.mp4"  # You can customize the filename and extension
            file.save(video_path)

            # Video processing logic
            video = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                # Preprocess the frame
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
                frame = (frame / 127.5) - 1  # Normalize the frame
                frames.append(frame)
            
            video.release()
            
            # Concatenate frames along the batch dimension
            video_data = np.concatenate(frames, axis=0)
            
            # Predictions for each frame
            predictions = model.predict(video_data)
            
            # Aggregate predictions over all frames
            avg_prediction = np.mean(predictions, axis=0)
            index = np.argmax(avg_prediction)
            class_name = class_names[index]
            confidence_score = avg_prediction[index] * 100
            
            # Example email sending
            subject = f"Prediction Result: {class_name[2:]}"
            message = f"The model predicted {class_name[2:]} with {confidence_score:.2f}% confidence."
            send_email('sender@example.com', ['poloce@example.com'], subject, message)
            
            # Generate graphs based on confidence score
            scatter_plot_img, bar_chart_img, pie_chart_img = generate_graphs(confidence_score)
            
            return render_template('index.html', class_name=class_name[2:], confidence_score=f"{confidence_score:.2f}%",
                                scatter_plot_img=scatter_plot_img, bar_chart_img=bar_chart_img, pie_chart_img=pie_chart_img)


        else:
            return render_template('index.html', message='Invalid file type selected')


if __name__ == '__main__':
    app.run(debug=True)
