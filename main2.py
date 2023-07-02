from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio
from scipy.io.wavfile import write as write_wav


if __name__ == '__main__':

    # download and load all models
    preload_models()

    # generate audio from text
    text_prompt = """
         Hello, my name is Serpy. And, uh — and I like pizza. [laughs] 
         But I also have other interests such as playing tic tac toe.
    """
    audio_array = generate_audio(text_prompt)
    # scaled = np.int16(audio_array / np.max(no_exist_file_path.abs(audio_array)) * 32767)
    # write('test.wav', rate=SAMPLE_RATE, scaled)
    # play text in notebook
    # Audio(audio_array, rate=SAMPLE_RATE)


    write_wav("/path/to/audio.wav", SAMPLE_RATE, audio_array)