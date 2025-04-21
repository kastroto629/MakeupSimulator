import os
import torch
import cv2
import time
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import CSD_MT_eval

def get_makeup_transfer_results256(non_makeup_img, makeup_img, alpha_eye, alpha_eyebrow, alpha_lip, alpha_all, regions):

    # Alpha 값을 영역별로 매핑
    alpha_values = {
        "eye": alpha_eye,
        "eyebrow": alpha_eyebrow,
        "lip": alpha_lip,
        "all": alpha_all,
    }

    # 메이크업 전이 수행
    transfer_img = CSD_MT_eval.makeup_transfer256(non_makeup_img, makeup_img, alpha_values, regions)
    return transfer_img



example = {}
non_makeup_dir = 'examples/non_makeup'
makeup_dir = 'examples/makeup'
non_makeup_list = [os.path.join(non_makeup_dir, file) for file in os.listdir(non_makeup_dir)]
non_makeup_list.sort()
makeup_list = [os.path.join(makeup_dir, file) for file in os.listdir(makeup_dir)]
makeup_list.sort()

# Gradio 인터페이스 정의
with gr.Blocks() as demo:
    with gr.Group():
        with gr.Tab("CSD-MT"):
            with gr.Row():
                with gr.Column():
                    # Non-makeup 및 Makeup 이미지 업로드
                    non_makeup = gr.Image(source='upload', elem_id="image_upload", type="numpy",
                                          label="Non-makeup Image")
                    gr.Examples(non_makeup_list, inputs=[non_makeup], label="Examples - Non-makeup Image",
                                examples_per_page=6)

                    makeup = gr.Image(source='upload', elem_id="image_upload", type="numpy",
                                      label="Makeup Image")
                    gr.Examples(makeup_list, inputs=[makeup], label="Examples - Makeup Image", examples_per_page=6)

                with gr.Column():
                    image_out = gr.Image(label="Output", type="numpy", elem_id="output-img").style(height=550)

                    # 영역별 투명도 슬라이더
                    alpha_eye = gr.Slider(0, 1, value=1, step=0.1, label="Eye Makeup Transparency (Alpha)")
                    alpha_eyebrow = gr.Slider(0, 1, value=1, step=0.1, label="Eyebrow Makeup Transparency (Alpha)")
                    alpha_lip = gr.Slider(0, 1, value=1, step=0.1, label="Lip Makeup Transparency (Alpha)")
                    alpha_all = gr.Slider(0, 1, value=1, step=0.1, label="Overall Makeup Transparency (Alpha)")

                    # 체크박스 그룹 (영역 선택)
                    region_selector = gr.CheckboxGroup(
                        ["eye", "eyebrow", "lip", "all"],
                        label="Select Makeup Regions",
                        value=["all"],
                    )

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Apply Makeup! (CSD-MT)").style()

            # Click result
            btn.click(
                fn=get_makeup_transfer_results256,
                inputs=[
                    non_makeup,
                    makeup,
                    alpha_eye,
                    alpha_eyebrow,
                    alpha_lip,
                    alpha_all,
                    region_selector,
                ],
                outputs=image_out,
            )


demo.launch()

