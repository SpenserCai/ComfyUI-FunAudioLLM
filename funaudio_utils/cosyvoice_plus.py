'''
Author: SpenserCai
Date: 2024-10-04 14:21:08
version: 
LastEditors: SpenserCai
LastEditTime: 2024-10-04 16:07:20
Description: file content
'''
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import logging
from tqdm import tqdm
import time

class CosyVoicePlus(CosyVoice):
    
    def inference_zero_shot_with_spkmodel(self,tts_text, spkmodel,stream=False, speed=1.0):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            tts_text_token, tts_text_token_len = self.frontend._extract_text_token(tts_text)
            spkmodel["text"] = tts_text_token
            spkmodel["text_len"] = tts_text_token_len
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**spkmodel, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()