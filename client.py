from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import torch

from datetime import datetime
print(f"start {datetime.now()}")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)
pipeline.to(torch.device("cuda"))
diarization_result = pipeline("data/afjiv.wav")
print(f"end {datetime.now()}")
