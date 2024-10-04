import folder_paths
import torch
import torchaudio
import os
import numpy as np
from funasr import AutoModel
from funaudio_utils.pre import FunAudioLLMTool
from funaudio_utils.download_models import download_sensevoice_small
from funasr.utils.postprocess_utils import rich_transcription_postprocess

fAudioTool = FunAudioLLMTool()

CATEGORY_NAME = "FunAudioLLM - SenseVoice"

folder_paths.add_model_folder_path("SenseVoice", os.path.join(folder_paths.models_dir, "SenseVoice"))

class SenseVoiceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "less_than_30s":("BOOLEAN",{
                    "default": True
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)

    FUNCTION="generate"

    def generate(self,audio, less_than_30s):
        sensevoice_code_path = os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI-FunAudioLLM/sensevoice/model.py")
        speech = audio["waveform"]
        source_sr = audio["sample_rate"]
        speech = speech.astype(np.float32) / np.iinfo(np.int16).max
        if len(speech.shape) > 1:
            input_wav = speech.mean(-1)
        if source_sr != 16000:
            print(f"audio_fs: {source_sr}")
            resampler = torchaudio.transforms.Resample(source_sr, 16000)
            speech_t = torch.from_numpy(input_wav).to(torch.float32)
            speech = resampler(speech_t[None, :])[0, :].numpy()
        _, model_dir = download_sensevoice_small()
        model_arg = {
                "input":speech,
                "cache":{},
                "language":"auto",
                "batch_size_s":60,
        }
        if not less_than_30s:
            model = AutoModel(model=model_dir,
                              trust_remote_code=True,
                              remote_code=sensevoice_code_path,
                              vad_model="fsmn-vad",
                              vad_kwargs={"max_single_segment_time": 30000},
                              device="cuda:0",
                    )
            model_arg["merge_vad"] = True
            model_arg["merge_length_s"] = 15
            
        else:
            model = AutoModel(model=model_dir,
                              trust_remote_code=True,
                              remote_code=sensevoice_code_path,
                              device="cuda:0",
                    )
        output = model.generate(**model_arg)
        return (rich_transcription_postprocess(output[0]["text"]),)