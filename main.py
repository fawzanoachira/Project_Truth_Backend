from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperTokenizer

# Load model directly

# processor = AutoProcessor.from_pretrained("fawzanaramam/Whisper-Small-Finetuned-on-Surah-Fatiha")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("fawzanaramam/Whisper-Small-Finetuned-on-Surah-Fatiha")


asr_pipe = pipeline(model="fawzanaramam/the-truth-amma-juz", generate_kwargs={"language": "arabic", "task": "transcribe", "output_scores":True})  # change to "your-username/the-name-you-picked"

# from transformers import WhisperForConditionalGeneration
# from transformers import WhisperFeatureExtractor
# from transformers import WhisperTokenizer
# from transformers import pipeline

# feature_extractor = WhisperFeatureExtractor.from_pretrained("fawzanaramam/Whisper-Small-Finetuned-on-Surah-Fatiha")
# tokenizer = WhisperTokenizer.from_pretrained("fawzanaramam/Whisper-Small-Finetuned-on-Surah-Fatiha", language="arabic", task="transcribe")

# model = WhisperForConditionalGeneration.from_pretrained("fawzanaramam/Whisper-Small-Finetuned-on-Surah-Fatiha")
# forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="arabic", task="transcribe")

# asr_pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     feature_extractor=feature_extractor,
#     tokenizer=tokenizer,
#     chunk_length_s=30,
#     stride_length_s=(4, 2)
# )

def transcribe(audio):
    # text = asr_pipe(audio)["text"]
    text = asr_pipe(audio)
    return {"transcription": text}

# print(transcribe("flutter_sound_example.wav"))
app = FastAPI()

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
        return transcribe(file.filename)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    # return {"message": f"Successfully uploaded {file.filename}"}