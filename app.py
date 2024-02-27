import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from authtoken import auth_token
import torch
from diffusers import StableDiffusionPipeline
import os
import threading  # Import threading module

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("BARD LI")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Use CPU
pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token).to(device)

# def show_processing():
#     lmain.configure(text="Processing...")
#     app.update()

def update_image(image_path):
    img = Image.open(image_path)
    img = ImageTk.PhotoImage(img)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference!
    os.remove(image_path)  # Delete the generated image file after displaying

def generate_image():

    try:
        with torch.inference_mode():
            # Generate images
            result = pipe(prompt.get(), guidance_scale=8.5).images[0]  # Adjusted access here

        image_path = 'generatedimage.png'
        result.save(image_path)
        app.after(0, update_image, image_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def generate():
    threading.Thread(target=generate_image).start()

trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
