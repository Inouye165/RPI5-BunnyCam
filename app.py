import io
from threading import Condition

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)

# CAM/DISP0 is usually index 0
picam2 = Picamera2(0)
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)

output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Pi Camera</title>
    <style>
      body { font-family: sans-serif; margin: 16px; }
      .wrap { max-width: 1100px; margin: 0 auto; }
      img { width: 100%; height: auto; border-radius: 12px; }
      .meta { margin-top: 10px; opacity: 0.8; font-size: 14px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h2>Raspberry Pi Camera Feed</h2>
      <img src="/stream.mjpg" alt="camera stream" />
      <div class="meta">Local MJPEG stream over Wi-Fi</div>
    </div>
  </body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(PAGE)

def gen_frames():
    while True:
        with output.condition:
            output.condition.wait()
            frame = output.frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")

@app.get("/stream.mjpg")
def stream():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
