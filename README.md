# Facial-Recognition-using-Haar-Cascade
This repository contains a facial recognition project that utilizes Haar Cascade classifiers for detecting faces in images. The project demonstrates how to detect and extract facial features using OpenCV, store the embeddings in a PostgreSQL database, and perform similarity searches to identify faces.

Overview
Haar Cascade is a popular object detection method used for identifying objects in images and videos. This project focuses on facial recognition by detecting faces, extracting embeddings, and storing them for future comparisons.

Setup
Prerequisites
Ensure you have the following installed:

Python 3.7+
OpenCV
imgbeddings
psycopg2-binary
Pillow
PostgreSQL
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/Godfathxx/Facial-Recognition-using-Haar-Cascade.git
cd Facial-Recognition-using-Haar-Cascade
Install dependencies:

bash
Copy code
pip install opencv-python imgbeddings psycopg2-binary Pillow
Set up the PostgreSQL database:

Replace the SERVICE URI with your PostgreSQL service URI in the script to connect to your database.

Running the Facial Recognition
1. Face Detection and Cropping:
The following script loads an image, detects faces using the Haar Cascade classifier, and crops each detected face:

makefile
Copy code
```python
# Importing the cv2 library
import cv2

# Loading the Haar Cascade algorithm file
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Loading the image path - replace with your image path
file_name = "/content/Friends.jpg"
img = cv2.imread(file_name, 0)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Detecting the faces
faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

# Cropping and saving each detected face
i = 0
for x, y, w, h in faces:
    cropped_image = img[y : y + h, x : x + w]
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(target_file_name, cropped_image)
    i += 1
```
2. Storing Face Embeddings in PostgreSQL:
This script calculates face embeddings using imgbeddings and stores them in a PostgreSQL database:

python
Copy code
```python
# Importing required libraries
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

# Connecting to the PostgreSQL database
conn = psycopg2.connect("postgres://your_service_uri")

for filename in os.listdir("stored-faces"):
    img = Image.open("stored-faces/" + filename)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()
```
3. Face Recognition:
To recognize a face and find the closest match from the stored embeddings:

python
Copy code
```python
from PIL import Image
from imgbeddings import imgbeddings
from IPython.display import Image, display

# Load and calculate embeddings for the face to be recognized
file_name = "/content/joey.jpg"  # Replace with your image path
img = Image.open(file_name)
ibed = imgbeddings()
embedding = ibed.to_embeddings(img)

cur = conn.cursor()
string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
for row in rows:
    display(Image(filename="stored-faces/"+row[0]))
cur.close()
```
4. Database Setup:
To set up the PostgreSQL database for storing face embeddings:

sql
Copy code
```bash
psql "postgres://your_service_uri"
DROP TABLE IF EXISTS pictures;
CREATE TABLE pictures (picture text PRIMARY KEY, embedding vector(768));
```
Usage
Face Detection: Detect faces in images and crop them for further processing.
Face Embedding Storage: Store the embeddings of detected faces in a PostgreSQL database.
Face Recognition: Recognize and identify faces by comparing new face embeddings against stored ones.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
OpenCV
PostgreSQL
Haar Cascade Classifier
