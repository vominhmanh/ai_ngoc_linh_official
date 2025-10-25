import os
# import gradio as gr
import requests
import base64
import time
import io
from loguru import logger
from pathlib import Path
from typing import Optional, Tuple, List
import json

# Configuration
class Config:
    def __init__(self):
        self.base_url = os.getenv("LITELLM_BASE_URL", "https://api.thucchien.ai")
        self.gemini_base_url = os.getenv("GEMINI_BASE_URL", "https://api.thucchien.ai/gemini/v1beta")
        self.api_key = os.getenv("LITELLM_API_KEY", "sk-ug1poexeERrICjPNvLSooQ")

config = Config()

# ==================== IMAGE GENERATION (Imagen 4) ====================
PROMPT1 = """
Generate an adorable and whimsical horse character, celebrating Tet 2026. The horse should be charming and lovable, incorporating traditional Vietnamese Lunar New Year elements such as cherry blossoms, kumquat trees, or lucky money envelopes (li xi). Emphasize a festive and prosperous atmosphere, suitable for a banking savings campaign. The style should be vibrant, friendly, and visually appealing, conveying a sense of good fortune and happiness.
"""
PROMPT1 = """

"""

# ==================== IMAGE EDITING (Gemini) ====================
def edit_image(
    image_file: str="Screenshot 2025-10-25 at 13.57.31.png",
    prompt: str=PROMPT1,
    aspect_ratio: str="1:1",
    base_url: str="https://api.thucchien.ai",
    api_key: str="sk-ug1poexeERrICjPNvLSooQ") -> Tuple[Optional[str], str]:
    """Edit image using Gemini via direct API"""
    try:
        url = f"{base_url}/gemini/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        # Build parts list
        parts = []

        # Add image if provided
        if image_file:
            with open(image_file, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Determine mime type
            if image_file.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_file.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"

            parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": image_b64
                }
            })

        # Add text prompt
        parts.append({"text": prompt})

        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": aspect_ratio
                }
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()

        data = response.json()
        image_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)

        # Save to temporary file
        temp_path = f"edited_image_{int(time.time())}.png"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        return temp_path, "✅ Image edited successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

print(edit_image(PROMPT1))
exit()

# ==================== VIDEO GENERATION ====================
def generate_video(prompt: str, image_file: Optional[str], negative_prompt: str, aspect_ratio: str, resolution: str,
                  duration: str, person_generation: str, model: str, base_url: str, api_key: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
    """Generate video using Veo (text-to-video or image-to-video)"""
    try:
        logger.info(f"Video generation - Model: {model}, Aspect: {aspect_ratio}, Res: {resolution}, Duration: {duration}s, Person: {person_generation}")
        logger.debug(f"Prompt: {prompt[:100]}... | Has image: {image_file is not None} | Negative: {negative_prompt[:50] if negative_prompt else 'None'}...")
        progress(0, desc="Starting video generation...")

        # Step 1: Initiate generation
        url = f"{base_url}/models/{model}:predictLongRunning"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        # Build instance with optional image
        instance = {
            "prompt": prompt
        }

        # If image is provided, add it as base64
        if image_file:
            with open(image_file, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # Determine mime type
            if image_file.lower().endswith('.png'):
                mime_type = "image/png"
            elif image_file.lower().endswith(('.jpg', '.jpeg')):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"

            instance["image"] = {
                "bytesBase64Encoded": image_b64,
                "mimeType": mime_type
            }

        # Build parameters
        parameters = {
            "aspectRatio": aspect_ratio,
            "durationSeconds": int(duration),
            "personGeneration": person_generation
        }

        # Add resolution only for Veo 3.x models (not supported in Veo 2)
        if not model.startswith("veo-2"):
            parameters["resolution"] = resolution

        # Add negative prompt if provided
        if negative_prompt and negative_prompt.strip():
            parameters["negativePrompt"] = negative_prompt.strip()

        payload = {
            "instances": [instance],
            "parameters": parameters
        }

        logger.debug(f"Request URL: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        # Check for errors before raising status
        if not response.ok:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                logger.error(f"API Error: {error_msg}")
                return None, f"❌ API Error: {error_msg}"
            except Exception:
                logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
                return None, f"❌ HTTP {response.status_code}: {response.text[:200]}"

        operation_name = response.json().get("name")
        if not operation_name:
            logger.error("No operation name returned")
            return None, "❌ Failed to start video generation - no operation name returned"

        logger.info(f"Operation started: {operation_name}")
        progress(0.2, desc="Video generation in progress...")

        # Step 2: Poll for completion
        operation_url = f"{base_url}/{operation_name}"
        start_time = time.time()
        max_wait = 600  # 10 minutes

        while time.time() - start_time < max_wait:
            response = requests.get(operation_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"Video error: {data['error']}")
                return None, f"❌ Error: {data['error']}"

            if data.get("done", False):
                logger.info("Video generation operation completed")

                # Check for RAI filter or other generation issues
                generate_video_response = data.get("response", {}).get("generateVideoResponse", {})

                # Check if video was filtered by RAI
                if generate_video_response.get("raiMediaFilteredCount", 0) > 0:
                    reasons = generate_video_response.get("raiMediaFilteredReasons", [])
                    error_msg = "Video generation blocked:\n" + "\n".join(f"- {reason}" for reason in reasons)
                    logger.warning(f"RAI filter triggered: {reasons}")
                    return None, f"❌ {error_msg}"

                # Check if generatedSamples exists
                generated_samples = generate_video_response.get("generatedSamples", [])
                if not generated_samples:
                    logger.error("No generated samples in response")
                    return None, "❌ Video generation completed but no video was generated. Please try modifying your prompt."

                logger.info("Video generation complete, downloading...")
                progress(0.8, desc="Downloading video...")
                video_uri = generated_samples[0]["video"]["uri"]

                # Step 3: Download video
                # Extract file ID from URI (e.g., "https://generativelanguage.googleapis.com/v1beta/files/3j6svp4106e7")
                if "files/" in video_uri:
                    file_id = video_uri.split("files/")[-1]
                    # Remove /gemini/v1beta from base_url and construct download URL
                    api_base = base_url.replace("/gemini/v1beta", "")
                    download_url = f"{api_base}/gemini/download/v1beta/files/{file_id}:download?alt=media"
                else:
                    download_url = video_uri

                logger.debug(f"Download URL: {download_url}")
                response = requests.get(download_url, headers=headers, stream=True, timeout=180)
                response.raise_for_status()

                temp_path = f"generated_video_{int(time.time())}.mp4"
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Video saved: {temp_path}")
                progress(1.0, desc="Complete!")
                return temp_path, "✅ Video generated successfully!"

            elapsed = int(time.time() - start_time)
            progress(0.2 + (0.6 * elapsed / max_wait), desc=f"Generating... ({elapsed}s)")
            time.sleep(10)

        logger.warning("Video generation timeout")
        return None, "⏰ Timeout: Video generation took too long"

    except Exception as e:
        logger.error(f"Video generation error: {str(e)}", exc_info=True)
        return None, f"❌ Error: {str(e)}"

# ==================== TEXT-TO-SPEECH (V1) ====================
def text_to_speech_v1(text: str, voice: str, base_url: str, api_key: str) -> Tuple[Optional[str], str]:
    """Convert text to speech using Gemini TTS (V1)"""
    try:
        url = f"{base_url}/gemini/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        audio_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]

        # Decode base64 audio (PCM format)
        audio_bytes = base64.b64decode(audio_data)

        # Save as WAV file
        temp_path = f"generated_audio_{int(time.time())}.wav"

        # Convert PCM to WAV format
        import wave
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_bytes)

        return temp_path, "✅ Audio generated successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==================== TEXT-TO-SPEECH (Gemini MULTI-SPEAKER) ====================
def text_to_speech_multi_speaker(text: str, speaker1_name: str, speaker1_voice: str,
                                  speaker2_name: str, speaker2_voice: str,
                                  base_url: str, api_key: str) -> Tuple[Optional[str], str]:
    """Convert text to speech with multiple speakers using Gemini TTS"""
    try:
        url = f"{base_url}/gemini/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {
                                "speaker": speaker1_name,
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {
                                        "voiceName": speaker1_voice
                                    }
                                }
                            },
                            {
                                "speaker": speaker2_name,
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {
                                        "voiceName": speaker2_voice
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        audio_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]

        # Decode base64 audio (PCM format)
        audio_bytes = base64.b64decode(audio_data)

        # Save as WAV file
        temp_path = f"generated_multi_audio_{int(time.time())}.wav"

        # Convert PCM to WAV format
        import wave
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_bytes)

        return temp_path, "✅ Multi-speaker audio generated successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==================== TEXT-TO-SPEECH (Gemini) ====================
def text_to_speech(text: str, voice: str, base_url: str, api_key: str) -> Tuple[Optional[str], str]:
    """Convert text to speech using /audio/speech endpoint"""
    try:
        url = f"{base_url}/audio/speech"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gemini-2.5-flash-preview-tts",
            "input": text,
            "voice": voice
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        # Save audio file
        temp_path = f"generated_audio_{int(time.time())}.mp3"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        return temp_path, "✅ Audio generated successfully!"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ==================== GRADIO INTERFACE ====================
def create_ui():
    """Create the Gradio interface"""

    with gr.Blocks(title="AI Thực Chiến - Playground") as demo:
        gr.Markdown("# 🚀 AI Thực Chiến - Playground")
        gr.Markdown("Interactive playground for Google Gemini API - Chat, Images, Videos, and Text-to-Speech by AI NGỌC LINH")

        # Configuration Section
        with gr.Accordion("⚙️ API Configuration", open=False):
            with gr.Row():
                base_url_input = gr.Textbox(
                    label="Chat API Base URL",
                    value=config.base_url,
                    placeholder="http://localhost:4000"
                )
                gemini_base_url_input = gr.Textbox(
                    label="Gemini API Base URL",
                    value=config.gemini_base_url,
                    placeholder="http://localhost:4000/gemini/v1beta"
                )
            api_key_input = gr.Textbox(
                label="API Key",
                value=config.api_key,
                type="password",
                placeholder="sk-1234 or your Google AI Studio API key"
            )

        # Tabs for different functionalities
        with gr.Tabs():
            # ===== CHAT TAB =====
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500, label="Chat History")
                        msg = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            lines=3
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Column(scale=1):
                        chat_model = gr.Dropdown(
                            choices=["gemini-2.5-flash", "gemini-1.5-pro"],
                            value="gemini-2.5-flash",
                            label="Model"
                        )
                        chat_temp = gr.Slider(0, 2, value=0.7, step=0.1, label="Temperature")
                        chat_max_tokens = gr.Slider(100, 8000, value=2000, step=100, label="Max Tokens")

                def respond(message, chat_history, model, temp, max_tok, base_url, api_key):
                    chat_history.append((message, ""))
                    for bot_message in chat_completion(message, chat_history[:-1], model, temp, max_tok, base_url, api_key):
                        chat_history[-1] = (message, bot_message)
                        yield "", chat_history
                    return "", chat_history

                msg.submit(respond, [msg, chatbot, chat_model, chat_temp, chat_max_tokens,
                                    base_url_input, api_key_input], [msg, chatbot])
                submit_btn.click(respond, [msg, chatbot, chat_model, chat_temp, chat_max_tokens,
                                          base_url_input, api_key_input], [msg, chatbot])
                clear_btn.click(lambda: None, None, chatbot, queue=False)

            # ===== IMAGE TAB =====
            with gr.Tab("🎨 Image Generation (Imagen 4)"):
                with gr.Row():
                    with gr.Column():
                        img_prompt = gr.Textbox(
                            label="Image Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=5
                        )
                        img_aspect = gr.Dropdown(
                            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3"],
                            value="1:1",
                            label="Aspect Ratio"
                        )
                        img_generate_btn = gr.Button("Generate Image", variant="primary")

                    with gr.Column():
                        img_output = gr.Image(label="Generated Image", type="filepath")
                        img_status = gr.Textbox(label="Status", lines=2)
                        img_download = gr.File(label="Download Image")

                def generate_and_display(prompt, aspect, base_url, api_key):
                    img_path, status = generate_image(prompt, aspect, base_url, api_key)
                    return img_path, status, img_path

                img_generate_btn.click(
                    generate_and_display,
                    [img_prompt, img_aspect, gemini_base_url_input, api_key_input],
                    [img_output, img_status, img_download]
                )

            # ===== IMAGE GENERATION (CHAT) TAB =====
            with gr.Tab("🎨 Image Generation (Gemini)"):
                with gr.Row():
                    with gr.Column():
                        chat_img_prompt = gr.Textbox(
                            label="Image Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=5
                        )
                        chat_img_generate_btn = gr.Button("Generate Image", variant="primary")

                    with gr.Column():
                        chat_img_output = gr.Image(label="Generated Image", type="filepath")
                        chat_img_status = gr.Textbox(label="Status", lines=2)
                        chat_img_download = gr.File(label="Download Image")

                def generate_and_display_chat(prompt, base_url, api_key):
                    img_path, status = generate_image_chat(prompt, base_url, api_key)
                    return img_path, status, img_path

                chat_img_generate_btn.click(
                    generate_and_display_chat,
                    [chat_img_prompt, base_url_input, api_key_input],
                    [chat_img_output, chat_img_status, chat_img_download]
                )

            # ===== IMAGE EDITING TAB =====
            with gr.Tab("✏️ Image Editing (Gemini)"):
                with gr.Row():
                    with gr.Column():
                        edit_img_input = gr.Image(
                            label="Input Image (Optional)",
                            type="filepath",
                            sources=["upload"]
                        )
                        edit_img_prompt = gr.Textbox(
                            label="Edit Prompt",
                            placeholder="Describe how you want to edit the image or generate a new one...",
                            lines=5
                        )
                        edit_img_aspect = gr.Dropdown(
                            choices=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3"],
                            value="9:16",
                            label="Aspect Ratio"
                        )
                        edit_img_generate_btn = gr.Button("Generate/Edit Image", variant="primary")
                        gr.Markdown("""
                        **Usage:**
                        - **With image**: Upload an image and describe the edits you want
                        - **Without image**: Just enter a prompt to generate a new image
                        """)

                    with gr.Column():
                        edit_img_output = gr.Image(label="Generated/Edited Image", type="filepath")
                        edit_img_status = gr.Textbox(label="Status", lines=2)
                        edit_img_download = gr.File(label="Download Image")

                def edit_and_display(image, prompt, aspect, base_url, api_key):
                    img_path, status = edit_image(prompt, image, aspect, base_url, api_key)
                    return img_path, status, img_path

                edit_img_generate_btn.click(
                    edit_and_display,
                    [edit_img_input, edit_img_prompt, edit_img_aspect, base_url_input, api_key_input],
                    [edit_img_output, edit_img_status, edit_img_download]
                )

            # ===== VIDEO TAB =====
            with gr.Tab("🎬 Video Generation"):
                gr.Markdown("### Text-to-Video or Image-to-Video")
                with gr.Row():
                    with gr.Column():
                        vid_prompt = gr.Textbox(
                            label="Video Prompt",
                            placeholder="Describe the video you want to generate...\nFor image-to-video: describe the motion/transformation",
                            lines=5
                        )
                        vid_negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="e.g., blurry, low quality, distorted",
                            lines=2
                        )
                        vid_image = gr.Image(
                            label="Input Image (Optional - for Image-to-Video)",
                            type="filepath",
                            sources=["upload"]
                        )
                        vid_model = gr.Dropdown(
                            choices=["veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.0-generate-001", "veo-3.0-fast-generate-001", "veo-2.0-generate-001"],
                            value="veo-3.0-generate-001",
                            label="Model"
                        )
                        with gr.Row():
                            vid_aspect = gr.Dropdown(
                                choices=["16:9", "9:16"],
                                value="16:9",
                                label="Aspect Ratio"
                            )
                            vid_resolution = gr.Dropdown(
                                choices=["720p", "1080p"],
                                value="720p",
                                label="Resolution",
                                info="1080p only for 16:9 (not supported in Veo 2)"
                            )
                        with gr.Row():
                            vid_duration = gr.Dropdown(
                                choices=["4", "5", "6", "8"],
                                value="6",
                                label="Duration (seconds)",
                                info="Veo 3.1/3: 4/6/8s | Veo 2: 5/6/8s"
                            )
                            vid_person_gen = gr.Dropdown(
                                choices=["allow_all", "allow_adult", "dont_allow"],
                                value="allow_all",
                                label="Person Generation",
                                info="Text-to-video: allow_all | Image-to-video: allow_adult"
                            )
                        vid_generate_btn = gr.Button("Generate Video (Takes ~5-10 min)", variant="primary")
                        gr.Markdown("""
                        ⚠️ **Notes:**
                        - Video generation can take 5-10 minutes
                        - **Text-to-Video**: Just enter a prompt (no image needed)
                        - **Image-to-Video**: Upload an image + describe the desired motion
                        - **Negative Prompt**: Describe what to avoid in the video
                        - **Duration**: Veo 3.1/3 supports 4/6/8s, Veo 2 supports 5/6/8s
                        - **Person Generation**: Text-to-video uses 'allow_all', Image-to-video uses 'allow_adult'
                        """)

                    with gr.Column():
                        vid_output = gr.Video(label="Generated Video")
                        vid_status = gr.Textbox(label="Status", lines=2)
                        vid_download = gr.File(label="Download Video")

                def generate_and_display_video(prompt, image, negative, aspect, resolution, duration, person_gen, model, base_url, api_key, progress=gr.Progress()):
                    vid_path, status = generate_video(prompt, image, negative, aspect, resolution, duration, person_gen, model, base_url, api_key, progress)
                    return vid_path, status, vid_path

                vid_generate_btn.click(
                    generate_and_display_video,
                    [vid_prompt, vid_image, vid_negative_prompt, vid_aspect, vid_resolution, vid_duration, vid_person_gen, vid_model, gemini_base_url_input, api_key_input],
                    [vid_output, vid_status, vid_download]
                )

            # ===== TTS TAB (V1) =====
            with gr.Tab("🔊 Text-to-Speech (V1)"):
                with gr.Row():
                    with gr.Column():
                        tts_v1_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5
                        )
                        tts_v1_voice = gr.Dropdown(
                            choices=[
                                "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
                                "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
                                "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                                "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
                                "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
                            ],
                            value="Kore",
                            label="Voice",
                            info="Zephyr (Bright), Puck (Upbeat), Charon (Informative), Kore (Firm), etc."
                        )
                        tts_v1_generate_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column():
                        tts_v1_output = gr.Audio(label="Generated Audio", type="filepath")
                        tts_v1_status = gr.Textbox(label="Status", lines=2)
                        tts_v1_download = gr.File(label="Download Audio")

                def generate_and_display_audio_v1(text, voice, base_url, api_key):
                    audio_path, status = text_to_speech_v1(text, voice, base_url, api_key)
                    return audio_path, status, audio_path

                tts_v1_generate_btn.click(
                    generate_and_display_audio_v1,
                    [tts_v1_text, tts_v1_voice, base_url_input, api_key_input],
                    [tts_v1_output, tts_v1_status, tts_v1_download]
                )

            # ===== TTS TAB =====
            with gr.Tab("🔊 Text-to-Speech (Gemini)"):
                with gr.Row():
                    with gr.Column():
                        tts_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5
                        )
                        tts_voice = gr.Dropdown(
                            choices=[
                                "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
                                "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
                                "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                                "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
                                "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
                            ],
                            value="Zephyr",
                            label="Voice",
                            info="Zephyr (Bright), Puck (Upbeat), Charon (Informative), Kore (Firm), etc."
                        )
                        tts_generate_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column():
                        tts_output = gr.Audio(label="Generated Audio", type="filepath")
                        tts_status = gr.Textbox(label="Status", lines=2)
                        tts_download = gr.File(label="Download Audio")

                def generate_and_display_audio(text, voice, base_url, api_key):
                    audio_path, status = text_to_speech(text, voice, base_url, api_key)
                    return audio_path, status, audio_path

                tts_generate_btn.click(
                    generate_and_display_audio,
                    [tts_text, tts_voice, base_url_input, api_key_input],
                    [tts_output, tts_status, tts_download]
                )

            # ===== TTS MULTI-SPEAKER TAB =====
            with gr.Tab("🔊 Text-to-Speech (Gemini Multi-Speaker)"):
                with gr.Row():
                    with gr.Column():
                        tts_multi_text = gr.Textbox(
                            label="Dialogue Script",
                            placeholder="Example:\nMake Speaker1 sound tired and bored, and Speaker2 sound excited and happy:\nSpeaker1: So... what's on the agenda today?\nSpeaker2: You're never going to guess!",
                            lines=8
                        )
                        with gr.Row():
                            with gr.Column():
                                speaker1_name = gr.Textbox(
                                    label="Speaker 1 Name",
                                    value="Joe",
                                    placeholder="e.g., Joe, Speaker1"
                                )
                                speaker1_voice = gr.Dropdown(
                                    choices=[
                                        "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
                                        "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
                                        "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                                        "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
                                        "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
                                    ],
                                    value="Kore",
                                    label="Speaker 1 Voice"
                                )
                            with gr.Column():
                                speaker2_name = gr.Textbox(
                                    label="Speaker 2 Name",
                                    value="Jane",
                                    placeholder="e.g., Jane, Speaker2"
                                )
                                speaker2_voice = gr.Dropdown(
                                    choices=[
                                        "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
                                        "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
                                        "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
                                        "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima",
                                        "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
                                    ],
                                    value="Puck",
                                    label="Speaker 2 Voice"
                                )
                        tts_multi_generate_btn = gr.Button("Generate Multi-Speaker Audio", variant="primary")
                        gr.Markdown("""
                        **Instructions:**
                        - Use speaker names in your script (e.g., "Speaker1: Hello" or "Joe: Hello")
                        - You can add direction/emotion instructions at the beginning
                        - Configure speaker names and voices below
                        """)

                    with gr.Column():
                        tts_multi_output = gr.Audio(label="Generated Audio", type="filepath")
                        tts_multi_status = gr.Textbox(label="Status", lines=2)
                        tts_multi_download = gr.File(label="Download Audio")

                def generate_and_display_multi_audio(text, s1_name, s1_voice, s2_name, s2_voice, base_url, api_key):
                    audio_path, status = text_to_speech_multi_speaker(
                        text, s1_name, s1_voice, s2_name, s2_voice, base_url, api_key
                    )
                    return audio_path, status, audio_path

                tts_multi_generate_btn.click(
                    generate_and_display_multi_audio,
                    [tts_multi_text, speaker1_name, speaker1_voice, speaker2_name, speaker2_voice,
                     base_url_input, api_key_input],
                    [tts_multi_output, tts_multi_status, tts_multi_download]
                )

        # Footer
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - **Chat**: Supports conversation history and multiple Gemini models
        - **Image**: Generate images with various aspect ratios
        - **Video**: Text-to-video OR image-to-video generation using Veo 3.0 (requires 5-10 minutes)
        - **TTS**: Convert text to speech with different voice options

        ### 🔧 Configuration:
        - Configure your LiteLLM base URLs and API key in the configuration section
        """)

    return demo

# ==================== MAIN ====================
if __name__ == "__main__":
    demo = create_ui()
    demo.queue()  # Enable queuing for video generation
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

