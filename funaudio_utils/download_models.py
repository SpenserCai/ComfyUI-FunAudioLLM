'''
Author: SpenserCai
Date: 2024-10-04 13:54:01
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 22:26:22
Description: file content
'''
import modelscope
import os
import folder_paths
from modelscope import snapshot_download

# Download the model
base_cosyvoice_model_path = os.path.join(folder_paths.models_dir, "CosyVoice")
base_sensevoice_model_path = os.path.join(folder_paths.models_dir, "SenseVoice")

def download_cosyvoice_300m(is_25hz=False):
    model_name = "CosyVoice-300M"
    model_id = "iic/CosyVoice-300M"
    if is_25hz:
        model_name = "CosyVoice-300M-25Hz"
        model_id = "iic/CosyVoice-300M-25Hz"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_cosyvoice_300m_sft(is_25hz=False):
    model_name = "CosyVoice-300M-SFT"
    model_id = "iic/CosyVoice-300M-SFT"
    if is_25hz:
        model_name = "CosyVoice-300M-SFT-25Hz"
        model_id = "MachineS/CosyVoice-300M-SFT-25Hz"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_sensevoice_small():
    model_name = "SenseVoiceSmall"
    model_id = "iic/SenseVoiceSmall"
    model_dir = os.path.join(base_sensevoice_model_path, model_name)
    snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def download_cosyvoice_300m_instruct():
    model_name = "CosyVoice-300M-Instruct"
    model_id = "iic/CosyVoice-300M-Instruct"
    model_dir = os.path.join(base_cosyvoice_model_path, model_name)
    snapshot_download(model_id=model_id, local_dir=model_dir)
    return model_name, model_dir

def get_speaker_default_path():
    return os.path.join(base_cosyvoice_model_path, "Speaker")