import tkinter as tk
from tkinter import ttk
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import pandas as pd
from PIL import Image, ImageTk
from tkinter import Label, Frame

features = ["Store", "IsHoliday","Dept", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Month", "Year", "Week", "Total_MarkDown", "Type", "Size"]
target = "Weekly_Sales"

# df = pd.read_csv("data.csv") # Replace "data.csv" with your data file name

train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(train_data[features], train_data[target])

# Evaluate the model on the validation set
val_preds = model.predict(val_data[features])
val_rmse = mean_squared_error(val_data[target], val_preds, squared=False)

def predict_sales():
    store = float(store_entry.get())
    isholiday = float(isholiday_entry.get())
    dept = float(dept_entry.get())
    temperature = float(temp_entry.get())
    fuel_price = float(fuel_entry.get())
    cpi = float(cpi_entry.get())
    unemployment = float(unemp_entry.get())
    total_markdown = float(markdown_entry.get())
    month = float(month_entry.get())
    year = float(year_entry.get())
    week = float(week_entry.get())
    store_type = float(type_entry.get())
    store_size = float(size_entry.get())

    input_data = pd.DataFrame({"Store": [store],"IsHoliday": [isholiday],"Dept":[dept],"Temperature": [temperature],
                               "Fuel_Price": [fuel_price],"CPI": [cpi],"Unemployment": [unemployment],"Month":[month],
                               "Year":[year], "Week":[week],"Total_MarkDown": [total_markdown], "Type":[store_type], "Size":[store_size]
    })
    pred_sales = model.predict(input_data)[0]

    # Create a new window for the output
    output_window = tk.Toplevel(root)
    output_window.title("Prediction Result")
    output_window.geometry("300x100")
    
    # Display the output label on the new window
    output_label = ttk.Label(output_window, text=f"Predicted Weekly Sales: {pred_sales:.2f}")
    output_label.pack(padx=10, pady=10)

# Set up the GUI
root = tk.Tk()
root.title("Walmart Weekly Sales Prediction")
root.geometry("1200x800")


image1 = Image.open("im.jpg")
bg_img = ImageTk.PhotoImage(image1, master=root)
bg_label = Label(root, image=bg_img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

title_label = tk.Label(root, text="Walmart Weekly Sales Prediction", font=("Georgia", 24, "bold"))
title_label.pack(pady=20)


input_frame = Frame(root)
# Define the input widgets
store_label = ttk.Label(input_frame, text="Store:")
store_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
store_entry = ttk.Entry(input_frame)
store_entry.grid(row=0, column=1, padx=10, pady=10,sticky="w")


isholiday_label = ttk.Label(input_frame, text="IsHoliday:")
isholiday_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
isholiday_entry = ttk.Entry(input_frame)
isholiday_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

dept_label = ttk.Label(input_frame, text="Dept:")
dept_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
dept_entry = ttk.Entry(input_frame)
dept_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

temp_label = ttk.Label(input_frame, text="Temperature:")
temp_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
temp_entry = ttk.Entry(input_frame)
temp_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

fuel_label = ttk.Label(input_frame, text="Fuel Price:")
fuel_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
fuel_entry = ttk.Entry(input_frame)
fuel_entry.grid(row=4, column=1, padx=10,pady=10, sticky="w")

cpi_label = ttk.Label(input_frame, text="CPI:")
cpi_label.grid(row=5, column=0, padx=10, pady=10, sticky="w")
cpi_entry = ttk.Entry(input_frame)
cpi_entry.grid(row=5, column=1, padx=10, pady=10, sticky="w")

unemp_label = ttk.Label(input_frame, text="Unemployment:")
unemp_label.grid(row=6, column=0, padx=10, pady=10, sticky="w")
unemp_entry = ttk.Entry(input_frame)
unemp_entry.grid(row=6, column=1, padx=10, pady=10, sticky="w")

markdown_label = ttk.Label(input_frame, text="Total Markdown:")
markdown_label.grid(row=7, column=0, padx=10, pady=10, sticky="w")
markdown_entry = ttk.Entry(input_frame)
markdown_entry.grid(row=7, column=1, padx=10, pady=10, sticky="w")

month_label = ttk.Label(input_frame, text="Month:")
month_label.grid(row=8, column=0, padx=10, pady=10, sticky="w")
month_entry = ttk.Entry(input_frame)
month_entry.grid(row=8, column=1, padx=10, pady=10, sticky="w")

year_label = ttk.Label(input_frame, text="Year:")
year_label.grid(row=9, column=0, padx=10, pady=10, sticky="w")
year_entry = ttk.Entry(input_frame)
year_entry.grid(row=9, column=1, padx=10, pady=10, sticky="w")

week_label = ttk.Label(input_frame, text="Week:")
week_label.grid(row=10, column=0, padx=10, pady=10, sticky="w")
week_entry = ttk.Entry(input_frame)
week_entry.grid(row=10, column=1, padx=10, pady=10, sticky="w")

type_label = ttk.Label(input_frame, text="Type:")
type_label.grid(row=11, column=0, padx=10, pady=10, sticky="w")
type_entry = ttk.Entry(input_frame)
type_entry.grid(row=11, column=1, padx=10, pady=10, sticky="w")

size_label = ttk.Label(input_frame, text="Size:")
size_label.grid(row=12, column=0, padx=10, pady=10, sticky="w")
size_entry = ttk.Entry(input_frame)
size_entry.grid(row=12, column=1, padx=10, pady=10, sticky="w")

predict_button = ttk.Button(input_frame, text="Predict Sales", command=predict_sales)
predict_button.grid(row=14, column=0, columnspan=2, padx=10, pady=10)

# Pack the input frame
input_frame.pack(pady=50)

# Place the input frame in the center of the root window
input_frame.place(relx=0.5, rely=0.5, anchor="center")
root.mainloop()
