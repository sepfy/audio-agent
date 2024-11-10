import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import threading
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
import librosa
from faster_whisper import WhisperModel

model_size = 'base'
logger = logging.getLogger('pc')
pcs = set()
task = None
MAX_AUDIO_BUFFER_SIZE = 16000 * 20
ROOT = os.path.dirname(__file__)


class VoiceAgent(threading.Thread):
    def __init__(self, step=500, duration=5000):
        super().__init__()
        self.sample_rate = 16000
        self.step = step
        self.duration = duration
        self.step_size = int(self.sample_rate * self.step / 1000)
        self.duration_size = int(self.sample_rate * self.duration / 1000)
        self.audio_buffer = np.ndarray(shape=(0, ))
        self.stop_event = threading.Event()
        self.model = WhisperModel(model_size, device='cpu', compute_type='int8', cpu_threads=8)
        self.lock = threading.Lock()
        self.segment_id = 0

    def add_frames_to_buffer(self, frames):
        with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, frames])
            #print(self.audio_buffer.shape)

    async def process_audio(self, data_channel):
        while True:
            audio_chunk = None
            audio_buffer_size = 0
            last_processed_size = 0
            with self.lock:
                if self.audio_buffer.size - last_processed_size > self.step_size:
                    last_processed_size = self.audio_buffer.size
                    audio_chunk = self.audio_buffer

                if self.audio_buffer.size > MAX_AUDIO_BUFFER_SIZE:
                    print('Buffer size is too large...')
            
            if audio_chunk is not None:
                segments, info = self.model.transcribe(audio_chunk, language='en', beam_size=5)
                for segment in segments:
                    result = {
                        'segment_id': self.segment_id,
                        'text': segment.text
                    }
                    if (data_channel.readyState == 'open'):
                        data_channel.send(json.dumps(result))
            else:
                await asyncio.sleep(0.1)

            if self.audio_buffer.size > self.duration_size:
                with self.lock:
                    self.audio_buffer = self.audio_buffer[self.duration_size - self.step_size * 4:]
                    last_processed_size = self.audio_buffer.size
                    self.segment_id += 1

            await asyncio.sleep(0.1)

voice_agent = VoiceAgent()

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

def filter_sdp_for_pcma(sdp):
    lines = sdp.splitlines()
    filtered_lines = []
    for line in lines:
        if line.startswith("a=rtpmap") and "PCMA" not in line:
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines) + '\n'

async def whip(request):

    sdp = await request.text()

    offer = RTCSessionDescription(sdp=sdp, type='offer')

    pc = RTCPeerConnection()

    data_channel = pc.createDataChannel('chat')

    @data_channel.on('message')
    def on_message(message):
        print('Data channel message:', message)

    @data_channel.on('open')
    async def on_open():
        print('Data channel open')
        global task
        if task is not None:
            task.cancel()
        task = asyncio.create_task(voice_agent.process_audio(data_channel))

    @data_channel.on('close')
    def on_close():
        print('Data channel closed')
        task.cancel()

    @pc.on('connectionstatechange')
    async def on_connectionstatechange():
        log_info('Connection state is %s', pc.connectionState)
        if pc.connectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

        if pc.connectionState == 'connected':
            print('Peer connection is connected...')

    pc_id = 'PeerConnection(%s)' % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + ' ' + msg, *args)

    log_info('Created for %s', request.remote)

    @pc.on('track')
    async def on_track(track):
        log_info('Track %s received', track.kind)

        if track.kind == 'audio':
            while True:
                try:
                    frame = await track.recv()
                    audio = frame.to_ndarray().reshape(-1)
                    audio = audio.astype(np.float32) / 32768.0
                    audio_resampled = librosa.resample(audio, orig_sr=8000, target_sr=16000)
                    voice_agent.add_frames_to_buffer(audio_resampled)
                except Exception as e:
                    print(e)
                    break

        @track.on('ended')
        async def on_ended():
            log_info('Track %s ended', track.kind)

    await pc.setRemoteDescription(offer)
    transceiver = pc.getTransceivers()[0]
    codecs = transceiver.receiver.getCapabilities('audio').codecs
    pcma_codec = []
    for codec in codecs:
        if codec.mimeType == 'audio/PCMA':
            pcma_codec.append(codec)
    transceiver.setCodecPreferences(pcma_codec)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    sdp = filter_sdp_for_pcma(pc.localDescription.sdp)
    return web.Response(
        status=201,
        content_type='application/sdp',
        text=sdp,
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WebRTC speech-to-text server'
    )
    parser.add_argument('--cert-file', help='SSL certificate file (for HTTPS)')
    parser.add_argument('--key-file', help='SSL key file (for HTTPS)')
    parser.add_argument(
        '--host', default='0.0.0.0', help='Host for HTTP server (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', type=int, default=8080, help='Port for HTTP server (default: 8080)'
    )
    parser.add_argument('--verbose', '-v', action='count')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    simple_whip_server = web.Application()
    simple_whip_server.on_shutdown.append(on_shutdown)
    simple_whip_server.router.add_get('/', index)
    simple_whip_server.router.add_post('/whip', whip)
    web.run_app(
        simple_whip_server, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
