'''
Author: SpenserCai
Date: 2024-10-04 13:22:31
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 14:09:40
Description: file content
'''
import librosa
import torch
import torchaudio

class FunAudioLLMTool:
    def __init__(self):
        self.max_val = 0.8
        self.prompt_sr, self.target_sr = 16000, 22050

    def postprocess(self,speech, top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        speech = torch.concat([speech, torch.zeros(1, int(self.target_sr * 0.2))], dim=1)
        return speech
    
    def audio_resample(self, waveform, source_sr):
        waveform = waveform.squeeze(0)
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != self.prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=self.prompt_sr)(speech)
        return speech
    