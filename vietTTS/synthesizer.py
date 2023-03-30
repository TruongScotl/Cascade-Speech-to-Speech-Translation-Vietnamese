import re
import glob
import unicodedata
# from argparse import ArgumentParser
from pathlib import Path
from vietnam_number import n2w

import soundfile as sf

from hifigan.mel2wave import mel2wave
from nat.config import FLAGS
from nat.text2mel import text2mel

from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



# parser = ArgumentParser()
# parser.add_argument("--text", type=str)
# parser.add_argument("--output", default="clip.wav", type=Path)
# parser.add_argument("--sample-rate", default=16000, type=int)
# parser.add_argument("--silence-duration", default=-1, type=float)
# parser.add_argument("--lexicon-file", default=None)
# args = parser.parse_args()

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        do_sample=True,
        top_k=100,
        top_p=0.8,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

def nat_normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    sil = FLAGS.special_phonemes[FLAGS.sil_index]
    text = re.sub(r"[\n.,:]+", f" {sil} ", text)
    text = text.replace('"', " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,:;?!]+", f" {sil} ", text)
    text = re.sub("[ ]+", " ", text)
    text = re.sub(f"( {sil}+)+ ", f" {sil} ", text)
    temp = re.findall(r'\d+', text)
    for i in temp:
        text = text.replace(i,n2w(i))
    return text.strip()


# text = nat_normalize_text(args.text)
# print("Normalized text input:", text)
# mel = text2mel(text, args.lexicon_file, args.silence_duration)
# wave = mel2wave(mel)
# print("writing output to file", args.output)
# sf.write(str(args.output), wave, samplerate=args.sample_rate)



def syntheaudio(path, output, sample_rate, silence_duration, lexicon_file):
    audio_paths = path
    transcriptions = model.transcribe(audio_paths)
    en = transcriptions[0]['transcription']
    en_vi = translate_en2vi(en)
    text = nat_normalize_text(en_vi)
    print("Normalized text input:", text)
    mel = text2mel(text, lexicon_file, silence_duration)
    wave = mel2wave(mel)
    print("writing output to file", output)
    sf.write(str(output), wave, samplerate=sample_rate)

# path = ['/Users/macos/Desktop/Final_Report/Data/test_slice_data/source/train/4.wav']
# output = '/Users/macos/Documents/GitHub/vietTTS/assets/infore/clip1.wav'
sample_rate = 16000
silence_duration = 0.2
lexicon_file = '/Users/macos/Documents/GitHub/vietTTS/assets/infore/lexicon.txt'

def multisyn(base_path, output):
    order = 0
    list_path =  [f for f in glob.glob(base_path+"/*.wav")]
    for i in list_path:
        order +=1
        syntheaudio([i], output + '/' + str(order) + '.wav', sample_rate, silence_duration, lexicon_file)

path = '/Users/macos/Desktop/Final_Report/Data/test_slice_data/source/train'
output = '/Users/macos/Desktop/Final_Report/Data/test_slice_data/source/test_syn'
multisyn(path, output)



#syntheaudio(path, output, sample_rate, silence_duration, lexicon_file)
