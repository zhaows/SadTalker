from pydub import AudioSegment


def trans_audio_format(input_path, input_format, output_format):
    audio = AudioSegment.from_file(input_path, input_format)
    filename = input_path.split('.')[0]
    output_path = '%s.%s'%(filename, output_format)
    audio.export(output_path, format=output_format)
    return output_path

if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]
    input_format = input_path.split('.')[1]
    output_format = 'wav'
    output_path = trans_audio_format(input_path, input_format, output_format)
    print(output_path)
