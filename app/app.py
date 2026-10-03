import gradio as gr

def generate_video(image):
    """Placeholder. Once the adapter exists, this will call it and return
    an actual generated video.
    """
    if image is None:
        return None
    # TODO: replace with real call once trajectory_encoder + motion_adapter + LTX inference exist
    print("Received an image. Model not wired up yet.")
    return None


with gr.Blocks(title="Motion Controlled Video") as demo:
    gr.Markdown("# Motion Controlled Video")
    gr.Markdown("Upload an image, then (eventually) draw a trajectory and generate a video.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")
            # TODO: replace with a proper trajectory-drawing input once the
            # model side is ready (gr.Sketchpad or clickable points on the image)
            generate_btn = gr.Button("Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Output")

    generate_btn.click(fn=generate_video, inputs=image_input, outputs=video_output)


if __name__ == "__main__":
    demo.launch(share=True)
