'''
  For more samples please visit https://github.com/Azure-Samples/cognitive-services-speech-sdk 
'''
import time
import azure.cognitiveservices.speech as speechsdk
from multiprocessing import Process,Queue

# Creates an instance of a speech config with specified subscription key and service region.
speech_key = "a6ea17326d1044ca9d889c6faee1e365"
service_region = "eastus"

def tts(text, voice_name=''):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"

    # output
    #audio_config = speechsdk.audio.AudioOutputConfig(filename="output.wav")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_synthesizer.speak_text_async(text).get()
    #print(result)
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

class TTS(object):
    def __init__(self, voice_name='zh-CN-XiaoxiaoNeural'):
        self._voice_name = voice_name
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.speech_synthesis_voice_name = self._voice_name
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self._speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        self._speech_synthesizer.synthesis_completed.connect(self.speech_synthesizer_synthesis_completed_cb)
        self._is_synthesizering = False

    def speech_synthesizer_synthesis_completed_cb(self, evt: speechsdk.SessionEventArgs):
        self._is_synthesizering = False
        print('Speak into your microphone:')

    def start_tts_async(self, text):
        self._resultFuture = self._speech_synthesizer.speak_text_async(text)
        self._is_synthesizering = True

    def start_tts(self, text):
        self._result = self._speech_synthesizer.speak_text(text)

    def is_synthesizering(self):
        return self._is_synthesizering

    def stop_tts(self):
        self._speech_synthesizer.stop_speaking()
        self._is_synthesizering = False

class MultiTTS(object):
    def __init__(self):
        self._tts_putonghua = TTS()
        self._tts_yueyu = TTS('yue-CN-XiaoMinNeural')
        self._tts_wuyu = TTS('wuu-CN-XiaotongNeural')
        self._tts_sichuan = TTS('zh-CN-sichuan-YunxiNeural')
        self._tts = self._tts_putonghua

    def start_tts_async(self, text, voice_name='普通话'):
        if voice_name in ['普通话']:
            self._tts = self._tts_putonghua 
        elif voice_name in ['四川话', '西南官话']:
            self._tts = self._tts_sichuan
        elif voice_name in ['吴语']:
            self._tts = self._tts_wuyu 
        elif voice_name in ['广东话', '粤语']:
            self._tts = self._tts_yueyu
        else:
            pass
        self._tts.start_tts_async(text)

    def stop_tts(self):
        self._tts.stop_tts()

    def is_synthesizering(self):
        return self._tts.is_synthesizering()

def recognize_from_audio(audio_file):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language="zh-CN"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    q = Queue()
    speech_recognizer.recognized.connect(lambda evt : recognized_handler(evt, q=q))

    speech_recognition_result = speech_recognizer.start_continuous_recognition()

def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language="zh-CN"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def stop_cb(evt):
    print('CLOSING on {}'.format(evt))
    speech_recognizer.stop_continuous_recognition()
    global done
    done = True

def recognizing_handler(e : speechsdk.SpeechRecognitionEventArgs, q=Queue()):
    if speechsdk.ResultReason.RecognizingSpeech == e.result.reason and len(e.result.text) > 0 :
        if '停止' in e.result.text or '别放了' in e.result.text:
            q.put(e.result.text)
            print("Recognizing: {}".format(e.result.text))
            #print("Offset in Ticks: {}".format(e.result.offset))
            #print("Duration in Ticks: {}".format(e.result.duration))


def recognized_handler(e : speechsdk.SpeechRecognitionEventArgs, q=Queue()) :
    if speechsdk.ResultReason.RecognizedSpeech == e.result.reason and len(e.result.text) > 0 :
        q.put(e.result.text)
        #print(q.get())
        print("Recognized: {}".format(e.result.text))
        #print("Offset in Ticks: {}".format(e.result.offset))
        #print("Duration in Ticks: {}".format(e.result.duration))

done = False
def recognize_from_microphone_continuous(q):
    #print('recognize_from_microphone_continuous')
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language="zh-CN"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    #speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognizing.connect(lambda evt : recognizing_handler(evt, q=q))
    speech_recognizer.recognized.connect(lambda evt : recognized_handler(evt, q=q))
    #speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    #speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    #print("Speak into your microphone: ")
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)


if __name__ == '__main__':
    #q = Queue()
    #recognize_from_microphone_continuous(q)
    #tts('我爱北京天安门')
    #tts = TTS()
    #tts.start_tts_async('我爱就发了汕德卡卷发梳颗粒剂工卡打撒国际卡到啦估计索拉卡立卡时代峰峻噶第三课噶扩大捡垃圾老嘎打撒')
    #print(tts.is_synthesizering())
    #time.sleep(15)
    #print(tts.is_synthesizering())
    #tts.stop_tts()
    import sys
    audio_file = sys.argv[1]
    recognize_from_audio(audio_file)

