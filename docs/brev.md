# QazMeeting AI on NVIDIA Brev

This is a deployment recipe, not a claim that a Brev instance is running.
Provisioning a GPU instance can incur compute and storage charges. Check the
organization credits and instance price before creating one.

The Streamlit server, Ollama model, transcription and SQLite database all run
on the same Brev instance. Meeting audio is uploaded from the browser to that
instance; it is not processed on the viewer's laptop. Use Brev's authenticated
tunnel for access rather than opening port 8501 directly to the internet.

## Prepare the instance

Use a Linux x86_64 NVIDIA instance with enough RAM and disk for PyTorch,
Whisper, Ollama and model caches. In a persistent workspace, obtain this
repository and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-brev.txt
ollama pull qwen2.5:3b-instruct-q4_K_M
```

The current `faster-whisper` CUDA runtime requires compatible CUDA 12 cuBLAS
and cuDNN 9 libraries. Verify `nvidia-smi` and a short transcription on the
selected Brev image before exposing the app. The `small` multilingual model
downloads once and remains in the instance cache. Russian, Kazakh and mixed
speech are not forced to a single language.

## Start the application

Run Ollama in one process. In the application process, set:

```bash
export AUDIO_STT_BACKEND=faster-whisper
export AUDIO_STT_MODEL=small
export AUDIO_STT_DEVICE=cuda
export AUDIO_STT_COMPUTE_TYPE=float16
export OLLAMA_MODEL=qwen2.5:3b-instruct-q4_K_M
export QAZMEETING_DB_PATH="$PWD/data/meetings.sqlite3"

python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Keep the project and `data/` directory on the instance's persistent disk.
The database stores full transcripts and meeting results. Do not put its path
inside Git or share the database file. The interface offers a saved meetings
section for reopening, reviewing and exporting results.

In the Brev console, open the instance's Access tab and add port `8501` under
Using Tunnels. Brev provides an authenticated URL for the team. The application
does not implement a second user account system, so access to the tunnel and
the instance must be limited to authorized reviewers.

Speaker diarization needs access to the gated
`pyannote/speaker-diarization-community-1` model or a local model path. Without
it, transcription continues with `speaker_id="UNKNOWN"` and a warning.
The meeting agent can also reject every proposed task when evidence is weak;
review its warnings and the transcript before exporting.
