
from GE2E.speaker_encoder.models.lstm import LSTMSpeakerEncoder
from GE2E.speaker_encoder.speaker_encoder_config import SpeakerEncoderConfig
from utils.audio import AudioProcessor
import re
import json
import fsspec
import torch
import numpy as np
import argparse

def read_json(json_path):
    config_dict = {}
    try:
        with fsspec.open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError:
        # backwards compat.
        data = read_json_with_comments(json_path)
    config_dict.update(data)
    return config_dict


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data

def get_emb(path):
    use_cuda = False
    source_file = path
    # config
    config_dict = read_json("saved_models/config.json")
    # print(config_dict)

    # model
    config = SpeakerEncoderConfig(config_dict)
    config.from_dict(config_dict)

    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )

    speaker_encoder.load_checkpoint("saved_models/best_model.pth.tar", eval=True, use_cuda=use_cuda)

    # preprocess
    speaker_encoder_ap = AudioProcessor(**config.audio)
    # normalize the input audio level and trim silences
    speaker_encoder_ap.do_sound_norm = True
    speaker_encoder_ap.do_trim_silence = True

    # compute speaker embeddings

    # extract the embedding
    waveform = speaker_encoder_ap.load_wav(
        source_file, sr=speaker_encoder_ap.sample_rate
    )
    spec = speaker_encoder_ap.melspectrogram(waveform)
    spec = torch.from_numpy(spec.T)
    if use_cuda:
        spec = spec.cuda()
    spec = spec.unsqueeze(0)
    embed = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
    embed = embed.squeeze()
    return embed


if __name__ == '__main__':
    print(get_emb("/Volumes/Extend/AI/majo/sliced/ it's Motion.wav"))