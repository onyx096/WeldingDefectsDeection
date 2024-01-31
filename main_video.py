import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import numpy as np
import av
import cv2
import asyncio
import logging
import logging.handlers
import threading
import base64
from base64 import decodebytes
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)


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

сlasses_rus = {
        '0': "Пора",
        '1': "Подрез",
        '2': "Разрыв дуги",
        '3': "Трещина",
        '4': "Подрез",
        '5': "Натёк",
        '6': "Включение шлака",
        '7': "Непровар",
}

classes = {
        '0': "air-hole",
        '1': "bite-edge",
        '2': "broken-arc",
        '3': "crack",
        '4': "undercut",
        '5': "overlap",
        '6': "slag-inclusion",
        '7': "unfused",
}

IMAGE_SIZE = 720

st.set_page_config(
    page_title="Weld-Helper",
)


def main():
    st.header("Real-time Weld Defect Detection")

    defect_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f" {thread.name} ({thread.ident})")


def defect_detection():
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


    class RoboflowVideoProcessor(VideoProcessorBase):
        _overlap = ovrlp_thr
        _confidence = conf_thr

        def __init__(self) -> None:
            self._overlap = ovrlp_thr
            self._confidence = conf_thr

        def set_overlap_confidence(self, overlap, confidence):
            self._overlap = overlap
            self._confidence = confidence

        def _annotate_image(self, image, detections):
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

                    button_size = (text_width + 20, text_height + 20)
                    button_img = Image.new("RGBA", button_size, color)
                    button_draw = ImageDraw.Draw(button_img)
                    button_draw.text(
                        (10, 4), text, font=font, fill=(30, 30, 30, 30)
                    )
                    image.paste(button_img, (int(x1), int(y1 - (text_height + 20))))
            return np.asarray(image)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

            image = frame.to_ndarray(format="bgr24")
            height, width, channels = image.shape
            scale = IMAGE_SIZE / max(height, width)
            image = cv2.resize(image, (round(scale * width), round(scale * height)))
            rf = Roboflow(api_key="API_KEY")
            project = rf.workspace("denis-tkachenko-mzw2k").project("welding-defects-detection-wtxow")
            model = project.version('1').model
            preds = model.predict(image, confidence=conf_thr, overlap=ovrlp_thr).json()
            detections = preds['predictions']
            annotated_image = self._annotate_image(image, detections)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


    webrtc_ctx = webrtc_streamer(
            key="logo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=None,
            video_processor_factory=RoboflowVideoProcessor,
            async_processing=True,
        )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_overlap_confidence(
            ovrlp_thr, conf_thr
        )

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()





