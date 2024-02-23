import base64
import sys


def audio2base64(audio_file):
    output_str = ''
    with open(audio_file, 'rb') as fr:
        output_str = fr.read()
        output_str = base64.b64encode(output_str)
    return output_str 

def base64_to_audio(encode_str, output_path):
    text = base64.b64decode(encode_str)
    with open(output_path, 'wb') as fw:
        fw.write(text)
    return 

def main():
    audio_file = sys.argv[1]
    output_file = sys.argv[2]
    res = audio2base64(audio_file)
    base64_to_audio(res, output_file)
    pass

def audio_main():
    output_file = sys.argv[1]
    b64_str = ''
    base64_to_audio(b64_str, output_file)
if __name__ == "__main__":
    audio_main()
