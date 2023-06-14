import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
from googlenet import googleNet
from keras.optimizers import Adam

st.markdown("<H1 style='text-align: center;'>Klasifikasi Kualitas Biji Jagung</H1>", unsafe_allow_html=True)

with st.sidebar:
    col1, col2, col3 = st.columns([2,5,2])

    with col1:
        st.write('')

    with col2:
        st.image('img/logoutm.png', width=150)

    with col3:
        st.write(' ')
    
    st.markdown("<H2 style='text-align: center;'>Klasifikasi Kualitas Biji Jagung</H2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Firda Ayu Safitri 19-098</p>", unsafe_allow_html=True)
    
    selected = option_menu("Main Menu", ["Home", 'About', 'Klasifikasi'], 
        icons=['house', 'info', 'search'], default_index=0)
    selected

if selected == "Home":
    st.markdown("""<p style='text-align: justify;'>Aplikasi ini dapat digunakan untuk klasifikasi kualitas biji jagung. 
    Jagung (Zea Mays) merupakan salah satu tanaman pangan penghasil karbohidrat 
    dan merupakan makanan pokok kedua Warga Negara Indonesia setelah padi. 
    Pada penelitian ini, dataset yang digunakan adalah dataset kualitas biji jagung yang 
    diklasifikasikan menjadi 4 kelas yakni broken (biji jagung berbentuk tidak 
    beraturan), discolored (biji jagung berubah warna), pure (biji jagung sehat) dan 
    silkcut (biji jagung pecah).</p>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2,6,2])

    with col1:
        st.write('')

    with col2:
        st.image('img/jagung.jpg', use_column_width='auto', caption='src: disnakkan.blitarkab.go.id')

    with col3:
        st.write(' ')

elif selected == "About":
    st.markdown("""<p style='text-align: justify;'>Pada penelitian ini, klasifikasi kualitas biji jagung 
    menggunakan algoritma GoogLeNet. GoogLeNet merupakan salah satu arsitektur dari metode Convolutional 
    Neural Network (CNN).  Data yang digunakan dalam penelitian ini berbentuk citra 
    kualitas biji jagung dengan ekstensi .png dengan total data sebanyak 17.801 data citra. 
    Adapun jumlah data citra pada masing-masing kelas akan ditunjukkan pada tabel di bawah ini.</p>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        df = pd.DataFrame(
            columns=('Kelas','Jumlah'),
            data=[['Broken (Tidak beraturan)', 5670],['Discolored (Berubah warna)', 3115],['Pure (Sehat)', 7265],['Silkcut (Pecah)', 1751],['TOTAL', 17801]]
            )
        AgGrid(df)

    with col2:
        img = pd.DataFrame(
            columns=('Broken','Discolored','Pure','Silkcut'),
            data=[[5670,0,0,0],[0,3115,0,0],[0,0,7265,0],[0,0,0,1751]]
            )
        st.bar_chart(img, height=200)
    
else:
    st.markdown("""<p style='text-align: justify;'>Aplikasi ini dapat digunakan untuk klasifikasi kualitas biji jagung.
    Upload citra biji jagung di bawah ini dan klik hasil klasifikasi untuk melihat hasilnya.</p>""", unsafe_allow_html=True)

    def save_uploadedfile(uploadedfile):
        with open(os.path.join("img", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    st.subheader('Post a Picture!')
    img_file_buffer = st.file_uploader('Post a Picture!', type=['png', 'jpg', 'jpeg'])

    if img_file_buffer is not None:
        save_uploadedfile(img_file_buffer)
        col1, col2 = st.columns(2)

        with col1:
            gambar = st.button('Lihat Gambar')

        with col2:
            result = st.button("Hasil Klasifikasi")

        if gambar:
            st.image(img_file_buffer, width=224)

        if result:
            shape=(224,224,3)
            class_names=['Broken','Discolored','Pure','Silkcut']

            weights_path = 'model/model.h5'
            googlenet_pred = googleNet(shape, weights_path, use_top=False)

            path='img/'+img_file_buffer.name
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            img = np.array(img, dtype = 'float32')
            img = np.expand_dims(img, axis=0)

            opt = Adam(learning_rate=0.0001)
            googlenet_pred.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

            out = googlenet_pred.predict(img)
            
            predicted_label = np.argmax(out)
            predicted_class_name = class_names[predicted_label]
            st.success(f"{predicted_class_name}")
