'''
Author: SpenserCai
Date: 2024-10-04 12:13:28
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-05 12:23:01
Description: file content
'''
import os
import folder_paths
import numpy as np
import torch
from funaudio_utils.pre import FunAudioLLMTool
from funaudio_utils.download_models import download_cosyvoice_300m, get_speaker_default_path, download_cosyvoice_300m_sft,download_cosyvoice_300m_instruct
from funaudio_utils.cosyvoice_plus import CosyVoicePlus
from cosyvoice.utils.common import set_all_random_seed

fAudioTool = FunAudioLLMTool()

CATEGORY_NAME = "FunAudioLLM - CosyVoice"

folder_paths.add_model_folder_path("CosyVoice", os.path.join(folder_paths.models_dir, "CosyVoice"))

def return_audio(output,t0,spk_model):
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
    if spk_model is not None:
        return (audio,spk_model,)
    else:
        return (audio,)

from time import time as ttime
class CosyVoiceZeroShotNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
            },
            "optional":{
                "prompt_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPK_MODEL",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO","SPK_MODEL",)
    
    FUNCTION="generate"

    def generate(self, tts_text, speed, seed, use_25hz, prompt_text=None, prompt_wav=None, speaker_model=None):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        if speaker_model is None:
            assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
            speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
            prompt_speech_16k = fAudioTool.postprocess(speech)
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k,False,speed)
            spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k)
            del spk_model['text']
            del spk_model['text_len']
            return return_audio(output,t0,spk_model)
        else:
            print('get zero_shot inference request')
            print(model_dir)
            set_all_random_seed(seed)
            output = cosyvoice.inference_zero_shot_with_spkmodel(tts_text, speaker_model,False,speed)
            return return_audio(output,t0,speaker_model)

class CosyVoiceSFTNode:
    sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING",),
                "speaker_name":(s.sft_spk_list,{
                    "default":"中文女"
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, speaker_name, speed, seed, use_25hz):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m_sft(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, speaker_name, False, speed)
        return return_audio(output,t0,None)
    
class CosyVoiceCrossLingualNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, prompt_wav, speed, seed, use_25hz):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)
        cosyvoice = CosyVoicePlus(model_dir)
        speech = fAudioTool.audio_resample(prompt_wav["waveform"], prompt_wav["sample_rate"])
        prompt_speech_16k = fAudioTool.postprocess(speech)
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, False, speed)
        return return_audio(output,t0,None)

class CosyVoiceInstructNode:
    sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING",),
                "speaker_name":(s.sft_spk_list,{
                    "default":"中文女"
                }),
                "instruct_text":("STRING",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, speaker_name, instruct_text, speed, seed):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m_instruct()
        cosyvoice = CosyVoicePlus(model_dir)
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, speaker_name, instruct_text, False, speed)
        return return_audio(output,t0,None)

class CosyVoiceLoadSpeakerModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "speaker_name":("STRING",),
                "model_dir":("STRING",{ "default":get_speaker_default_path()}),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, speaker_name, model_dir):
        # 加载模型
        spk_model_path = os.path.join(model_dir, speaker_name + ".pt")
        assert os.path.exists(spk_model_path), "Speaker model is not exist"
        spk_model = torch.load(os.path.join(model_dir, speaker_name + ".pt"),map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return (spk_model,)
    
class CosyVoiceLoadSpeakerModelFromUrlNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model_url":("STRING",),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("SPK_MODEL",)
    FUNCTION="generate"

    def generate(self, model_url):
        # 下载模型
        spk_model = torch.hub.load_state_dict_from_url(model_url,map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return (spk_model,)
 
class CosyVoiceSaveSpeakerModelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "spk_model":("SPK_MODEL",),
                "speaker_name":("STRING",),
                "model_dir":("STRING",{ "default":get_speaker_default_path()}),
            }
        }
    
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION="generate"

    def generate(self, spk_model, speaker_name, model_dir):
        # 判断目录是否存在，不存在则创建
        print(f"saving speaker model {speaker_name} to {model_dir}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # 保存模型
        torch.save(spk_model, os.path.join(model_dir, speaker_name + ".pt"))
        return speaker_name + '.pt'
        
        