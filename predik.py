from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

kelas_label=['celana_panjang', 'celana_pendek', 'kemeja', 'skirt', 'sweater_and_jacket', 't-shirt']

img = image.load_img('contoh.jpg', target_size=(224,224))

# Ubah gambar ke dalam bentuk array numerik
img_array = image.img_to_array(img)

# Sesuaikan dimensi gambar untuk input model
img_array = np.expand_dims(img_array, axis=0)

model=load_model('model_coba4.h5')

prediksi=model.predict(img_array)

print(prediksi)
predicted_class = np.argmax(prediksi)

print("Predicted class:", predicted_class)

prediksi_label = kelas_label[predicted_class]

print("Predicted class:", prediksi_label)