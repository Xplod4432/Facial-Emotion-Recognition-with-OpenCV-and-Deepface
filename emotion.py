import cv2
from deepface import DeepFace
from pytube import YouTube

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def process_video(video_url, output_file='output_with_emotions.mp4'):

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Download video stream using pytube
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(file_extension='mp4').first().url
    
    # Open the YouTube video stream with OpenCV
    cap = cv2.VideoCapture(video_stream)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = gray_frame[y:y + h, x:x + w]
    
            # Resize the face ROI to match the input shape of the model
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    
            # Normalize the resized face image
            normalized_face = resized_face / 255.0
    
            # Reshape the image to match the input shape of the model
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
    
            # Predict emotions using the pre-trained model
            preds = model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]
    
            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
        # Write the processed frame to the output file
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Output video saved as {output_file}")


if __name__ == "__main__":
    # Argument parser for runtime parameters
    parser = argparse.ArgumentParser(description="Process a YouTube video for emotion detection.")
    parser.add_argument(
        "--url", 
        type=str, 
        default="https://www.youtube.com/watch?v=embYkODkzcs",  # Default URL
        help="YouTube video URL (default: Rick Astley's Never Gonna Give You Up)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_with_emotions.mp4",
        help="Output file name (default: output_with_emotions.mp4)"
    )
    args = parser.parse_args()

    # Process the video with the specified or default URL
    process_video(args.url, args.output)
