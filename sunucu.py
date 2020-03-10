import struct
from gevent.pool import Pool
from gevent.server import StreamServer
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
from PIL import Image
import pickle
import glob
from ByteArray import ByteArray
# to load images
import os
import zlib

global i
i = 0

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_FILENAME = "captcha_model.hdf5"

MODEL_LABELS_FILENAME = "model_labels.dat"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)
i = 5_000

global I
I = 0


def check_data(data):
    i = struct.unpack("!i", data[:4])[0]

    data = data[4:i + 4]
    try:
        dec = zlib.decompress(data)
    except:
        return None
    if i == len(data):
        return dec
    return None


def handle2(socket, address):
    try:
        handle(socket, address)
    except:
        pass


def handle(socket, address):

    b = b""
    l = 0
    while True:
        data = socket.recv(8192)
        if len(data) == 0 and check_data(b):
            b = check_data(b)
            break
        # l=len(data)
        b += data
        if check_data(b):
            b = check_data(b)
            break
        if len(data) < l:
            break
        l = len(data)
    global i
    i += 1
   # open("capts/"+str(i)+".bin", "wb").write(b)
    if False:
        print(len(data))
        data = b
        #img_array = np.asarray(b, dtype=np.uint8)
        i = struct.unpack("!i", data[:4])[0]
        print(i)
        data = data[4:i + 4]
        try:
            data = zlib.decompress(data)
        except:
            print("err")
            socket.sendall(struct.pack("!h", 5)+b"ERROR")
    else:
        data = b
    en, boy, pixelSayisi = struct.unpack("!HHH", data[:6])

    data = data[6:]
    print(en, boy, pixelSayisi)
    Pixeller = []
    # for x in range(0,len(data)-4,4):
    #	RGBint=struct.unpack("!i",data[x:x+4])[0]
    #	Blue =  RGBint & 255
    #	Green = (RGBint >> 8) & 255
    #	Red =   (RGBint >> 16) & 255
    #	Pixeller.append((Red,Green,Blue))
    x = 0
    p = ByteArray(data)
    img = Image.new('RGBA', (en, boy), 4294967295)
    pixels = img.load()
    for pix in range(pixelSayisi):
        if len(p.bytes) == 0:
            print("break")
            break
        renk = p.readUnsignedInt()
        pixels[pix % en, pix / en] = renk

    img = img.resize((en * 10, boy * 10), Image.ANTIALIAS)

    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    global I
    I += 1
    background.save(f"yerler/{I}.jpg", 'JPEG', quality=100)
    img = cv2.imread(f"yerler/{I}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,  51,40)
    #ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    height, width = img.shape
    w = int(width/4)-1
    harfler = ""
    acc = 0
    Boyut = 30
    for i in range(4):
        resim = img[0:height, i*w:(i+1)*w]
        resim = img[0:height, i*w+10:(i+1)*w+10]
        r = None  # ayir(resim)

        #gray = cv2.copyMakeBorder(resim, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        #ret, threshed_img =  cv2.threshold(resim, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # resim=threshed_img
       # break
        letter_image = resize_to_fit(resim, Boyut, Boyut)
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        pr = lb.inverse_transform(prediction)
        letter = pr[0]
        harfler += letter
    c = harfler
    print(c, address)
    os.rename(f"yerler/{I}.jpg", f"yerler/{I}_{c}.jpg")
    socket.sendall(struct.pack("!h", len(c))+c.encode("utf-8"))


pool = Pool(10000)  # do not accept more than 10000 connections
server = StreamServer(('0.0.0.0', 1234), handle2, spawn=pool)
server.serve_forever()
