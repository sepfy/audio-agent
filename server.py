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

from faster_whisper import WhisperModel

model_size = "large-v3"



ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

class AudioAgent(threading.Thread):
    def __init__(self, sample_rate = 8000, duration = 0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_size = int(sample_rate * duration)
        self.lock = threading.Lock()
        self.frames = []
        self.stop_event = threading.Event()
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def append_frame(self, frame):
        with self.lock:
            self.frames.extend(frame)
            print('frame appended:', len(self.frames))

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:
                if len(self.frames) >= self.frame_size:
                    audio = np.array(self.frames[:self.frame_size]).reshape(-1)
                    self.frames = self.frames[self.frame_size:]
                    print('audio shape:', audio.shape)
                    text = self.model.transcribe(audio)
                    print('transcribed text:', text)
            self.stop_event.wait(0.1)

audio_agent = AudioAgent()
audio_agent.start()

async def whip(request):

    sdp = await request.text()

    offer = RTCSessionDescription(sdp=sdp, type='offer')

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    async def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            while True:
                try:
                    frame = await track.recv()
                    # (160, 1)
                    audio = frame.to_ndarray().reshape(-1, 1)
                    audio_agent.append_frame(audio)
                except:
                    break

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return web.Response(
        status=201,
        content_type="application/sdp",
        text=pc.localDescription.sdp,
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC speech-to-text demo (server)"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
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

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/whip", whip)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
