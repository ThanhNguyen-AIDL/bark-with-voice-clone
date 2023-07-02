from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch
from hubert.hubert_manager import HuBERTManager

from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer


if __name__ == '__main__':

    device = 'cuda'  # or 'cpu'
    model = load_codec_model(use_gpu=True if device == 'cuda' else False)


    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()


    # Load the HuBERT model
    hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  #

    audio_filepath = 'audio.wav'  # the audio you want to clone (under 13 seconds)
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)