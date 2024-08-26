import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
# read in data
data = pd.read_csv('C:\\Users\\Windows\\Desktop\\Major Project\\Stock project\\spy.csv')
 
# convert 'Date' column to datetime data type
data['Date'] = pd.to_datetime(data['Date'])
 
# Line plot of Open, Close, High, Low over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Date'], data['Open'], label='Open')
ax.plot(data['Date'], data['Close'], label='Close')
ax.plot(data['Date'], data['High'], label='High')
ax.plot(data['Date'], data['Low'], label='Low')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock Prices over Time')
ax.legend()
plt.show()
 
# Scatter plot of Open vs Close prices
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(data['Open'], data['Close'], alpha=0.5)
ax.set_xlabel('Open Price')
ax.set_ylabel('Close Price')
ax.set_title('Open vs Close Prices')
plt.show()
 
# Histogram of daily price changes
price_changes = data['Close'] - data['Open']
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(price_changes, bins=30)
ax.set_xlabel('Price Change')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Daily Price Changes')
plt.show()
 
# Heatmap of correlation matrix
corr_matrix = data[['Open', 'Close', 'High', 'Low']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm')
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(['Open', 'Close', 'High', 'Low'])
ax.set_yticklabels(['Open', 'Close', 'High', 'Low'])
ax.set_title('Correlation Matrix')
plt.colorbar(im)
plt.show()
 
data.isna().sum()
#Engulfing pattern signals
import random
def Revsignal1(data1):
    length = len(data1)
    High = list(data1['High'])
    Low = list(data1['Low'])
    Close = list(data1['Close'])
    Open = list(data1['Open'])
    signal = [0] * length
    bodydiff = [0] * length
 
    for row in range(1, length):
        bodydiff[row] = abs(Open[row]-Close[row])
        bodydiffmin = 0.003
        if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            Open[row-1]<Close[row-1] and
            Open[row]>Close[row] and 
            #Open[row]>=Close[row-1] and Close[row]<Open[row-1]):
            (Open[row]-Close[row-1])>=+0e-5 and Close[row]<Open[row-1]):
            signal[row] = 1
        elif (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
            Open[row-1]>Close[row-1] and
            Open[row]<Close[row] and 
            #Open[row]<=Close[row-1] and Close[row]>Open[row-1]):
            (Open[row]-Close[row-1])<=-0e-5 and Close[row]>Open[row-1]):
            signal[row] = 2
        else:
            signal[row] = 0
        #signal[row]=random.choice([0, 1, 2])
        #signal[row]=1
    return signal
data['signal1'] = Revsignal1(data)
data[data['signal1']==1].count()
 
#Target
def mytarget(data1, barsfront):
    length = len(data1)
    High = list(data1['High'])
    Low = list(data1['Low'])
    Close = list(data1['Close'])
    Open = list(data1['Open'])
    trendcat = [None] * length
    
    piplim = 300e-5
    for line in range (0, length-1-barsfront):
        for i in range(1,barsfront+1):
            if ((High[line+i]-max(Close[line],Open[line]))>piplim) and ((min(Close[line],Open[line])-Low[line+i])>piplim):
                trendcat[line] = 3 # no trend
            elif (min(Close[line],Open[line])-Low[line+i])>piplim:
                trendcat[line] = 1 #-1 downtrend
                break
            elif (High[line+i]-max(Close[line],Open[line]))>piplim:
                trendcat[line] = 2 # uptrend
                break
            else:
                trendcat[line] = 0 # no clear trend  
    return trendcat
 
data['Trend'] = mytarget(data,3)
#data.head(30)
 
import numpy as np
conditions = [(data['Trend'] == 1) & (data['signal1'] == 1),(data['Trend'] == 2) & (data['signal1'] == 2)]
values = [1, 2]
data['result'] = np.select(conditions, values)
 
trendId=2
print(data[data['result']==trendId].result.count()/data[data['signal1']==trendId].signal1.count())
data[ (data['Trend']!=trendId) & (data['signal1']==trendId) ] # false positives
 
dfpl = data[150:200]
import plotly.graph_objects as go
from datetime import datetime
 
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['Open'],
                high=dfpl['High'],
                low=dfpl['Low'],
                close=dfpl['Close'])])
# fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
#                 marker=dict(size=5, color="MediumPurple"),
#                 name="signal1")
fig.show()
 
 
# Set the directory path where the images are located
dir_path = '/content'
 
# Loop through the range of i values and delete the corresponding files
for i in range(127):
    file_path = os.path.join(dir_path, f'image_{i}.png')
    if os.path.exists(file_path):
        os.remove(file_path)
# Load the dataset
data = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/spy.csv')  # Assuming stock prices are stored in a CSV file
num_entries = data.shape[0]
num_parts = num_entries // 60  # Number of parts to divide the dataset into
 
# Convert the dataset into images
images = []
for i in range(num_parts):
    start_idx = i * 60
    end_idx = (i + 1) * 60
    part_data = data.iloc[start_idx:end_idx + 1]
    part_data = part_data[['Open', 'Close', 'High', 'Low']].values
    fig, ax = plt.subplots()
    ax.plot(part_data[:, 0], 'r', label='Open')
    ax.plot(part_data[:, 1], 'g', label='Close')
    ax.plot(part_data[:, 2], 'b', label='High')
    ax.plot(part_data[:, 3], 'k', label='Low')
    ax.legend()
    plt.savefig(f'image_{i}.png')
    plt.close(fig)
    images.append(f'image_{i}.png')
print(len(images))
 
# Prepare the data for CNN training
X = []
y = []
for i in range(num_parts):
    img_path = images[i]
    img = plt.imread(img_path)
    X.append(img)
    if data.iloc[(i+1)*60-1]['Close'] > data.iloc[(i+1)*60]['Close']:
        y.append(1)
    else:
        y.append(0)
import os
 
# Set the directory path where the images are located
dir_path = '/content'
 
# Loop through the range of i values and delete the corresponding files
for i in range(127):
    file_path = os.path.join(dir_path, f'image_{i}.png')
    if os.path.exists(file_path):
        os.remove(file_path)
X = np.array(X)
y = np.array(y)
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=47)
# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
 
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Train the model
model.fit(X_train, y_train, epochs=8, batch_size=32)
model.save_weights('model_weights.h5')
 
# Load the saved weights
model.load_weights('model_weights.h5')
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
 
# Make predictions using the trained model
input_data = data.iloc[-60:].values  # Assuming input data consists of the last 60 days of stock prices
fig, ax = plt.subplots()
ax.plot(input_data[:, 0], 'r', label='Open')
ax.plot(input_data[:, 1], 'g', label='Close')
ax.plot(input_data[:, 2], 'b', label='High')
ax.plot(input_data[:, 3], 'k', label='Low')
ax.legend()
plt.savefig('input_image.png')
plt.close(fig)
input_image = plt.imread('input_image.png').reshape(1, X.shape[1], X.shape[2], X.shape[3])
prediction = model.predict(input_image)
history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
# Plot the stock price prediction for the last 60 days
input_data = data.iloc[-60:].values
fig, ax = plt.subplots()
ax.plot(input_data[:, 0], 'r', label='Open')
ax.plot(input_data[:, 1], 'g', label='Close')
ax.plot(input_data[:, 2], 'b', label='High')
ax.plot(input_data[:, 3], 'k', label='Low')
ax.legend()


import matplotlib.pyplot as plt
import numpy as np

def convert_to_image(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data['Date'], data['Open'], label='Open')
    ax.plot(data['Date'], data['Close'], label='Close')
    ax.plot(data['Date'], data['High'], label='High')
    ax.plot(data['Date'], data['Low'], label='Low')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.title('Stock Price Data')
    plt.tight_layout()
    
    # Save the plot as an image
    plt.savefig('temp_image.png')
    plt.close(fig)
    
    # Load and return the saved image as a NumPy array
    image = plt.imread('temp_image.png')
    return image

 
def predict_movement(input_data):
    input_images = []
    for i in range(7):
        start_idx = -65 + i
        end_idx = -6 + i
        part_data = input_data[start_idx:end_idx + 1]
        input_image = convert_to_image(part_data)
        input_images.append(input_image)
    input_images = np.array(input_images)
 
    # Predict the movement for each day
    predictions = model.predict(input_images)
 
    # Convert the predictions to binary values (1 for up, 0 for down)
    binary_predictions = [1 if pred < 0.5 else 0 for pred in predictions]
 
    # Return the list of binary predictions
    return binary_predictions
 
input_data = data.iloc[-65:-7].values
binary_predictions = predict_movement(input_data)
print(binary_predictions)
actual_values=y[-7:]
days = [1, 2, 3, 4, 5,6,7]
fig, ax = plt.subplots()
ax.plot(days, binary_predictions, 'bo', markersize=10, fillstyle='none', label='Predicted')
ax.plot(days, actual_values, 'rs', markersize=15, fillstyle='none', label='Actual')
 
# Set x-axis and y-axis labels
ax.set_xlabel('Days')
ax.set_ylabel('Movement')
plt.xticks([1, 2, 3, 4, 5,6,7,8,9,10])
plt.yticks([0, 1])
# Set title
ax.set_title('Predicted vs Actual Movement')
 
# Add legend to the top right corner
ax.legend(loc='upper right')
 
plt.show()

if prediction > 0.5:
    print('Price goes up!')
else:
    print('Price goes down!')
