import tkinter as tk
from tkinter import filedialog, messagebox, font
from PIL import Image, ImageTk
import os
from backend_logic import TrafficDiffusionModel

class TrafficGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Forecasting for Intelligent Transport Systems")
        self.root.geometry("1100x700")
        
        self.model = TrafficDiffusionModel()
        
        # Determine background image path
        self.bg_path = "background.png"
        
        self.setup_ui()

    def setup_ui(self):
        # Background Image Setup
        if os.path.exists(self.bg_path):
            self.bg_image = Image.open(self.bg_path)
            self.bg_image = self.bg_image.resize((1100, 700), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(self.root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.root.configure(bg='#2b2b2b')

        # Header Frame (Pink background)
        header_frame = tk.Frame(self.root, bg="#FFEEEE", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_font = font.Font(family="Times New Roman", size=18, weight="bold")
        title_label = tk.Label(
            header_frame, 
            text="Traffic Prediction using Intelligent Transportation Systems with Big Data and Diffusion Models", 
            font=title_font, 
            bg="#FFEEEE", 
            fg="#CC00AA"
        )
        title_label.pack(pady=15)

        # Left control zone
        button_font = font.Font(family="Times New Roman", size=11, weight="bold")
        
        btn_y_start = 80
        btn_spacing = 50

        # Upload Dataset Button
        self.btn_upload = tk.Button(self.root, text="Upload Dataset", font=button_font, command=self.action_upload, width=20, bg="white")
        self.btn_upload.place(x=20, y=btn_y_start)

        # Data Preprocessing Button
        self.btn_preprocess = tk.Button(self.root, text="Data Preprocessing", font=button_font, command=self.action_preprocess, width=20, bg="white")
        self.btn_preprocess.place(x=250, y=btn_y_start)

        # Generate Model Button
        self.btn_train = tk.Button(self.root, text="Generate Diffusion Traffic Model", font=button_font, command=self.action_train_model, width=30, bg="white")
        self.btn_train.place(x=20, y=btn_y_start + btn_spacing)

        # Output/Predict Button
        self.btn_predict = tk.Button(self.root, text="Test Model & Predict Traffic Output", font=button_font, command=self.action_predict, width=30, bg="white")
        self.btn_predict.place(x=300, y=btn_y_start + btn_spacing)

        # Graph Button
        self.btn_graph = tk.Button(self.root, text="Accuracy & Loss Graphs", font=button_font, command=self.action_graph, width=22, bg="white")
        self.btn_graph.place(x=20, y=btn_y_start + (btn_spacing * 2))

        # Text Console Box for output
        text_font = font.Font(family="Arial", size=12, weight="bold")
        self.text_output = tk.Text(self.root, width=50, height=20, font=text_font, bg="white", fg="black", bd=2, relief="groove")
        self.text_output.place(x=20, y=280)
        
        self.log_message("Welcome to the Intelligent Transport Systems Simulator.")

    def log_message(self, message):
        self.text_output.insert(tk.END, message + "\n\n")
        self.text_output.see(tk.END)

    def action_upload(self):
        file_path = filedialog.askopenfilename(
            title="Select METR-LA Dataset",
            filetypes=(("H5 Dataset", "*.h5"), ("All files", "*.*"))
        )
        if file_path:
            self.log_message(f"Selected Dataset: {os.path.basename(file_path)}")
            success, msg = self.model.load_dataset(file_path)
            self.log_message(msg)

    def action_preprocess(self):
        success, msg = self.model.preprocess_data()
        self.log_message(msg)

    def action_train_model(self):
        success, msg = self.model.generate_models()
        self.log_message(msg)
        if success:
            acc = self.model.get_diffusion_accuracy()
            self.log_message(f"Diffusion Training Model Accuracy = {acc}%")

    def action_predict(self):
        # Simulating the prediction / output matrix display mapped to "Predict Traffic"
        success, msg = self.model.show_confusion_matrix()
        if not success:
            self.log_message(msg)

    def action_graph(self):
        success, msg = self.model.show_accuracy_graphs()
        if not success:
            self.log_message(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficGUI(root)
    root.mainloop()
