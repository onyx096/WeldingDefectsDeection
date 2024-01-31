import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import numpy as np
import os
from shutil import rmtree
import zipfile
import time



color_map = {
        '0': "#E23D28", #air-hole
        '1': "#FFBF00", #bite-edge
        '2': "#FF8200", #broken-arc
        '3': "#FFEF00", #crack
        '4': "#CB410B", #undercut
        '5': "#ED1B24", #overlap
        '6': "#F5F5DC", #slag-inclusion
        '7': "#FF3800", #unfused
    }

classes_rus = {
        '0': "Пора",
        '1': "Подрез",
        '2': "Разрыв дуги",
        '3': "Трещина",
        '4': "Подрез",
        '5': "Натёкmain_video.py",
        '6': "Включение шлака",
        '7': "Непровар",
}

classes = {
        '0': "Air-hole",
        '1': "Bite-edge",
        '2': "Broken-arc",
        '3': "Crack",
        '4': "Undercut",
        '5': "Overlap",
        '6': "Slag-inclusion",
        '7': "Unfused",
}


def classify(image):
    rf = Roboflow(api_key='API_KEY')
    project = rf.workspace("denis-tkachenko-mzw2k").project("welding-defects-detection-wtxow")
    model = project.version('1').model

    # infer on a local image
    return model.predict(image, confidence=conf_thr, overlap=ovrlp_thr).json()


def annotate_image(image, detections):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=25)

    for box in detections:
        color = color_map[box["class"]]
        x1 = box["x"] - box["width"] / 2
        x2 = box["x"] + box["width"] / 2
        y1 = box["y"] - box["height"] / 2
        y2 = box["y"] + box["height"] / 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if True:
            text = classes[box["class"]]
            left, top, right, bottom = font.getbbox(text)
            text_width = right - left
            text_height = bottom - top

            # set button size + 10px margins
            button_size = (text_width + 20, text_height + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text(
                (10, 4), text, font=font, fill=(30, 30, 30, 30)
            )
            # put button on source image in position (0, 0)
            image.paste(button_img, (int(x1), int(y1-(text_height + 20))))
    return np.asarray(image)



st.set_page_config(
    page_title="Weld-Helper",
)

st.title('Weld defects detection')


with st.sidebar:
    uploaded_file = st.file_uploader("Select an image to upload", type=["png", "jpg", "jpeg", 'zip'],
                                        accept_multiple_files=False)
conf_thr = st.sidebar.slider(
    "Confidence threshold:",
    0,
    100,
    10,
    5,
)
ovrlp_thr = st.sidebar.slider(
    "Overlap threshold:",
    0,
    100,
    30,
    5,
)

help_url = 'https://drive.google.com/file/d/1onHkfxWpe5RBdKzOtUgRvkmkzkYESPj5/view?usp=share_link'

st.sidebar.write("###")
st.sidebar.link_button('How it works', url=help_url ,use_container_width=True)

init_dir = os.getcwd()

with st.spinner('Doing my best...'):
    if uploaded_file is not None:
        if str(uploaded_file.name).endswith('.zip'):

            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with zipfile.ZipFile(uploaded_file.name, 'r') as zip_ref:
                zip_ref.extractall()

            os.chdir(f'{os.getcwd()}/{uploaded_file.name}'[:-4])

            i = 0
            for file in os.listdir():
                image = Image.open(file)
                uploaded_img = np.array(image)
                inferenced_img = uploaded_img.copy()
                preds = classify(inferenced_img)
                detections = preds['predictions']

                st.write(f'Image: :red[{file}], created :red[{time.ctime(os.path.getctime(file))}]')

                if len(detections) == 0:
                    st.image(inferenced_img, use_column_width=True)
                    st.success('Defects are not detected in the weld')
                else:
                    annotated_image = annotate_image(inferenced_img, detections)
                    image_comparison(
                        img1=inferenced_img,
                        img2=annotated_image,
                        starting_position=1,
                        show_labels=False,
                        make_responsive=True,
                        in_memory=True,
                    )
                    st.error('Defects are detected in the weld')
                i += 1
                if i != len(os.listdir()):
                    st.divider()

            os.chdir(init_dir)
            os.remove(uploaded_file.name)
            rmtree(f'{uploaded_file.name}'[:-4])
            rmtree('__MACOSX')

        else:
            image = Image.open(uploaded_file)
            uploaded_img = np.array(image)
            inferenced_img = uploaded_img.copy()

            preds = classify(inferenced_img)
            detections = preds['predictions']

            if len(detections) == 0:
                st.image(inferenced_img, use_column_width=True)
                st.success('Defects are not detected in the weld')
            else:
                annotated_image = annotate_image(inferenced_img, detections)
                image_comparison(
                    img1=inferenced_img,
                    img2=annotated_image,
                    starting_position=1,
                    show_labels=False,
                    make_responsive=True,
                    in_memory=True,
                )
                st.error('Defects are detected in the weld')