import whisper
def speech_to_command():
    model = whisper.load_model("medium.en")
    result = model.transcribe("/home/shyuni5/file/CORL2024/Sembot/real_bot/perception/speech_command.wav")
    print("Speech: ", result["text"])
    return result["text"].strip().lower()







