import cv2
import mediapipe as mp
import numpy as np
import torch
import tkinter as tk
from tkinter import messagebox, ttk
import os
from PIL import Image, ImageTk
import csv
from datetime import datetime
import time

class HandSignApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Sign Recognition")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Set default class names
        self.class_names = ["open", "close", "pointer", "ok","fuck"]
        
        # Load the trained model
        try:
            self.model = self.load_model()
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_loaded = False
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.root.destroy()
            return
        
        self.current_frame = None
        
        # Create GUI elements
        self.create_gui()
        
        # Training mode variables
        self.training_mode = False
        self.current_class = 0
        self.samples_collected = 0
        self.samples_needed = 50
        self.last_sample_time = 0
        self.sample_interval = 0.1  # seconds between samples
        
        # Start the video feed
        self.update_frame()
        
    def load_model(self):
        # Load your trained model with original architecture
        model = torch.nn.Sequential(
            torch.nn.Linear(42, 20),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),  # 4 output classes
            torch.nn.Softmax(dim=-1)
        )
        
        if not os.path.exists('F:\TTCS\model\keypoint_classifier\best_model.pth'):
            raise FileNotFoundError("Model file 'F:\TTCS\model\keypoint_classifier\best_model.pth' not found")
        
        # Load the model state
        state_dict = torch.load('F:\TTCS\model\keypoint_classifier\best_model.pth')
        print("Model state keys:", state_dict.keys())
        
        # Check if the model architecture matches the state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        # Print model summary
        print("Model architecture:")
        print(model)
        
        return model
    
    def create_gui(self):
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create video display
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack()
        
        # Create control buttons
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(pady=10)
        
        self.recognition_btn = tk.Button(
            self.control_frame,
            text="Start Recognition",
            command=self.toggle_recognition,
            state='normal' if self.model_loaded else 'disabled'
        )
        self.recognition_btn.pack(side=tk.LEFT, padx=5)
        
        self.training_btn = tk.Button(
            self.control_frame,
            text="Start Training",
            command=self.toggle_training
        )
        self.training_btn.pack(side=tk.LEFT, padx=5)
        
        # Create class selection for training
        self.class_frame = tk.Frame(self.main_frame)
        self.class_frame.pack(pady=5)
        
        self.class_label = tk.Label(self.class_frame, text="Select Class:")
        self.class_label.pack(side=tk.LEFT)
        
        # Create dropdown for class selection
        self.class_var = tk.StringVar()
        self.class_var.set(self.class_names[0])  # Set default to first class
        self.class_dropdown = ttk.Combobox(
            self.class_frame,
            textvariable=self.class_var,
            values=self.class_names,
            state="readonly",
            width=10
        )
        self.class_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Create progress bar for training
        self.progress_frame = tk.Frame(self.main_frame)
        self.progress_frame.pack(pady=5)
        self.progress = ttk.Progressbar(
            self.progress_frame,
            orient="horizontal",
            length=200,
            mode="determinate"
        )
        self.progress.pack()
        
        # Create status label
        self.status_label = tk.Label(self.main_frame, text="Status: Ready")
        self.status_label.pack(pady=5)
        
        # Create class names display
        self.class_names_frame = tk.Frame(self.main_frame)
        self.class_names_frame.pack(pady=5)
        self.class_names_label = tk.Label(
            self.class_names_frame,
            text="\n".join([f"Class {i}: {name}" for i, name in enumerate(self.class_names)])
        )
        self.class_names_label.pack()
        
        # Add button to edit class names
        self.edit_names_btn = tk.Button(
            self.class_names_frame,
            text="Edit Class Names",
            command=self.edit_class_names
        )
        self.edit_names_btn.pack(pady=5)
        
    def edit_class_names(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Class Names")
        
        names = []
        for i, name in enumerate(self.class_names):
            frame = tk.Frame(edit_window)
            frame.pack(pady=5)
            tk.Label(frame, text=f"Class {i}:").pack(side=tk.LEFT)
            entry = tk.Entry(frame)
            entry.insert(0, name)  # Set current name as default
            entry.pack(side=tk.LEFT)
            names.append(entry)
        
        def save_names():
            self.class_names = [entry.get() or f"Class {i}" for i, entry in enumerate(names)]
            self.update_class_names_display()
            # Update dropdown values
            self.class_dropdown['values'] = self.class_names
            edit_window.destroy()
        
        tk.Button(edit_window, text="Save", command=save_names).pack(pady=10)
        
    def update_class_names_display(self):
        if hasattr(self, 'class_names'):
            text = "\n".join([f"Class {i}: {name}" for i, name in enumerate(self.class_names)])
            self.class_names_label.config(text=text)
        
    def toggle_recognition(self):
        self.training_mode = False
        self.status_label.config(text="Status: Recognition Mode")
        self.progress['value'] = 0
        
    def toggle_training(self):
        try:
            # Get class index from the selected class name
            selected_class = self.class_var.get()
            self.current_class = self.class_names.index(selected_class)
            self.training_mode = True
            self.samples_collected = 0
            self.progress['value'] = 0
            self.status_label.config(
                text=f"Status: Training Mode - {selected_class} - Samples: 0/{self.samples_needed}"
            )
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        
    def process_hand_landmarks(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks and normalize them
                landmarks = []
                # Get wrist position for normalization
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for landmark in hand_landmarks.landmark:
                    # Normalize coordinates relative to wrist
                    x = landmark.x - wrist_x
                    y = landmark.y - wrist_y
                    landmarks.extend([x, y])
                
                # Convert to numpy array and normalize
                landmarks = np.array(landmarks, dtype=np.float32)
                
                # Print raw landmarks for debugging
                print("Raw landmarks:", landmarks)
                
                # Normalize the entire array
                landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
                
                # Print normalized landmarks for debugging
                print("Normalized landmarks:", landmarks)
                
                return landmarks
        
        return None
    
    def save_training_data(self, landmarks):
        current_time = time.time()
        if current_time - self.last_sample_time < self.sample_interval:
            return
            
        self.last_sample_time = current_time
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
            
        # Save landmarks to CSV
        csv_path = 'F:\TTCS\model\keypoint_classifier\keypoint.csv'
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_class] + landmarks.tolist())
            
        self.samples_collected += 1
        progress = (self.samples_collected / self.samples_needed) * 100
        self.progress['value'] = progress
        
        self.status_label.config(
            text=f"Status: Training Mode - {self.class_names[self.current_class]} - Samples: {self.samples_collected}/{self.samples_needed}"
        )
        
        if self.samples_collected >= self.samples_needed:
            messagebox.showinfo("Training Complete", f"Collected {self.samples_needed} samples for {self.class_names[self.current_class]}")
            self.training_mode = False
            self.status_label.config(text="Status: Ready")
            self.progress['value'] = 0
    
    def predict_gesture(self, landmarks):
        with torch.no_grad():
            input_tensor = torch.from_numpy(landmarks).float()
            output = self.model(input_tensor)
            
            # Normalize the output to get proper probabilities
            output = torch.nn.functional.softmax(output, dim=0)
            
            # Print raw output for debugging
            print("Raw model output:", output)
            
            predicted_class = torch.argmax(output).item()
            confidence = output[predicted_class].item()
            
            # Print all class confidences
            confidences = {name: output[i].item() for i, name in enumerate(self.class_names)}
            print("Class confidences:", confidences)
            
            # Get class name
            class_name = self.class_names[predicted_class]
            
            # Only return prediction if confidence is above threshold (lowered from 0.5 to 0.3)
            if confidence > 0.3:  # Lowered threshold
                return predicted_class, confidence, class_name
            else:
                return None, confidence, "Unknown"
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process hand landmarks
            landmarks = self.process_hand_landmarks(frame)
            
            if landmarks is not None:
                if self.training_mode:
                    # Save training data
                    self.save_training_data(landmarks)
                elif self.model_loaded:
                    # Predict gesture
                    predicted_class, confidence, class_name = self.predict_gesture(landmarks)
                    if predicted_class is not None:
                        cv2.putText(frame, f"{class_name} ({confidence:.2f})",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the video label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule the next update
        self.root.after(10, self.update_frame)
    
    def run(self):
        self.root.mainloop()
        
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandSignApp()
    app.run() 
        