import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from transformers import CLIPProcessor, CLIPModel
from models import ClipCapModel
import os


class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning with VLM")
        self.root.geometry("800x600")

        # Initialize models as None
        self.clip_model = None
        self.caption_model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Image Captioning with VLM",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Image selection button
        self.select_button = ttk.Button(main_frame, text="Select Image",
                                       command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=(0, 10), sticky=tk.W)

        # Selected file label
        self.file_label = ttk.Label(main_frame, text="No image selected",
                                   foreground="gray")
        self.file_label.grid(row=1, column=1, sticky=(tk.W, tk.E))

        # Generate caption button
        self.generate_button = ttk.Button(main_frame, text="Generate Caption",
                                         command=self.generate_caption,
                                         state=tk.DISABLED)
        self.generate_button.grid(row=1, column=2, padx=(10, 0), sticky=tk.E)

        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Selected Image", padding="10")
        self.image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S),
                             pady=(20, 10))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Image label
        self.image_label = ttk.Label(self.image_frame, text="No image selected",
                                    foreground="gray")
        self.image_label.grid(row=0, column=0)

        # Caption frame
        caption_frame = ttk.LabelFrame(main_frame, text="Generated Caption", padding="10")
        caption_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E),
                          pady=(10, 0))
        caption_frame.columnconfigure(0, weight=1)

        # Caption text widget
        self.caption_text = tk.Text(caption_frame, height=3, wrap=tk.WORD,
                                   font=("Arial", 12))
        self.caption_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Scrollbar for caption text
        caption_scrollbar = ttk.Scrollbar(caption_frame, orient=tk.VERTICAL,
                                         command=self.caption_text.yview)
        caption_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.caption_text.configure(yscrollcommand=caption_scrollbar.set)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E),
                       pady=(10, 0))

        # Store image path
        self.current_image_path = None

    def load_models(self):
        """Load the CLIP and caption models"""
        try:
            self.status_var.set("Loading models...")
            self.root.update()

            # Load CLIP model
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()

            # Load caption model
            model_path = "best_model.pth"
            if os.path.exists(model_path):
                self.caption_model = ClipCapModel(clip_dim=512)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.caption_model.load_state_dict(checkpoint['model_state_dict'])
                self.caption_model.to(self.device)
                self.caption_model.eval()
                self.status_var.set("Models loaded successfully")
            else:
                self.status_var.set("Warning: best_model.pth not found. Please train the model first.")
                messagebox.showwarning("Model Not Found",
                                     "best_model.pth not found. Please train the model first.")

        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)

    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )

        if file_path:
            self.current_image_path = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="black")
            self.display_image(file_path)

            # Enable generate button if models are loaded
            if self.caption_model is not None:
                self.generate_button.config(state=tk.NORMAL)

    def display_image(self, image_path):
        """Display the selected image in the GUI"""
        try:
            # Open and resize image for display
            image = Image.open(image_path)

            # Calculate display size (max 400x300)
            max_width, max_height = 400, 300
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # Convert to PhotoImage for tkinter
            photo = ImageTk.PhotoImage(image)

            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference

        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)

    def extract_clip_features(self, image_path):
        """Extract CLIP features from an image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features

    def generate_caption(self):
        """Generate caption for the selected image"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if self.caption_model is None:
            messagebox.showerror("Model Error", "Caption model is not loaded.")
            return

        try:
            self.status_var.set("Generating caption...")
            self.root.update()

            # Extract CLIP features
            image_features = self.extract_clip_features(self.current_image_path)

            # Generate caption
            captions = self.caption_model.generate(
                image_features,
                max_length=20,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_beams=3
            )

            # Display caption
            caption = captions[0] if captions else "No caption generated"
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(1.0, caption)

            self.status_var.set("Caption generated successfully")

        except Exception as e:
            error_msg = f"Error generating caption: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
