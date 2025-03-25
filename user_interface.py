import tkinter as tk
from tkinter import messagebox
import model_module



def display_number():
    try:
        num = int(entry.get())  # Get the number from entry field
        p_milage=model_module.predict_milage(num)
        label_output.config(text=f"Predicted Milage: {p_milage[0]}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number")

# Create the main window
root = tk.Tk()
root.title("Mileage Calculator for fuel efficiency")
root.geometry("500x300")

# Create input label and entry field
label = tk.Label(root, text="Enter a Engine Displacement:")
label.pack(pady=5)
entry = tk.Entry(root)
entry.pack(pady=5)

# Create submit button
button = tk.Button(root, text="Submit", command=display_number)
button.pack(pady=5)

# Create output label
label_output = tk.Label(root, text="")
label_output.pack(pady=5)

# Run the main loop
root.mainloop()
