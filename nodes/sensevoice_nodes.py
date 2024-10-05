import folder_paths
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
                "use_fast_mode":("BOOLEAN",{
                    "default": False
                }),
                "punc_segment":("BOOLEAN",{
                    "default": False
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)

    FUNCTION="generate"

    def generate(self,audio, use_fast_mode,punc_segment):
        sensevoice_code_path = os.path.join(folder_paths.base_path,"custom_nodes/ComfyUI-FunAudioLLM/sensevoice/model.py")
        speech = audio["waveform"]
        source_sr = audio["sample_rate"]
        speech = fAudioTool.audio_resample(speech, source_sr)
        speech = fAudioTool.postprocess(speech)
        _, model_dir = download_sensevoice_small()
        model_arg = {
                "input":speech[0],
                "cache":{},
                "language":"auto",
                "batch_size_s":60,
        }
        model_use_arg = {
            "model":model_dir,
            "trust_remote_code":True,
            "remote_code":sensevoice_code_path,
            "device":"cuda:0",
        }

        if not use_fast_mode:
            model_use_arg["vad_model"] = "fsmn-vad"
            model_use_arg["vad_kwargs"] = {"max_single_segment_time":30000}

            model_arg["merge_vad"] = True
            model_arg["merge_length_s"] = 15

        if punc_segment:
            model_use_arg["punc_model"] = "ct-punc-c"
        
        model = AutoModel(**model_use_arg)
        output = model.generate(**model_arg)
        return (rich_transcription_postprocess(output[0]["text"]),)