'''
Author: SpenserCai
Date: 2024-10-04 12:13:28
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 14:44:32
Description: file content
'''
import os
import folder_paths
import random
import numpy as np
import torch
from utils.pre import FunAudioLLMTool
from utils.download_models import download_cosyvoice_300m
from cosyvoice.cli.cosyvoice import CosyVoice

fAudioTool = FunAudioLLMTool()

CATEGORY_NAME = "FunAudioLLM"

folder_paths.add_model_folder_path("CosyVoice", os.path.join(folder_paths.models_dir, "CosyVoice"))

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def return_audio(output,t0):
    output_list = []
    for out_dict in output:
        output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
        output_numpy = output_numpy.astype(np.int16)
        # if speed > 1.0 or speed < 1.0:
        #     output_numpy = speed_change(output_numpy,speed,target_sr)
        output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
    t1 = ttime()
    print("cost time \t %.3f" % (t1-t0))
    audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":fAudioTool.target_sr}
    return (audio,)

from time import time as ttime
class CosyVoiceZeroShotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "prompt_wav": ("AUDIO",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "inference_mode":(["3s极速克隆","跨语种克隆"],{
                    "default": "3s极速克隆"
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
            },
            "optional":{
                "prompt_text":("TEXT",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    
    FUNCTION="generate"

    def generate(self, tts_text, speed, inference_mode, seed, use_25hz, prompt_text=None, prompt_wav=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoice(model_dir)
        if inference_mode == "3s极速克隆":
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
        speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
        prompt_speech_16k = fAudioTool.postprocess(speech)
        if inference_mode == "3s极速克隆":
            print('get zero_shot inference request')
            print(self.model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k,False,speed)
        elif inference_mode == '跨语种复刻':
            print('get cross_lingual inference request')
            print(self.model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k,False,speed)
        return return_audio(output,t0)
        

        