import os
import re
import tempfile
from typing import Tuple, List
from src.models import Paper, slugify
from src import console as con
from src.parsers.base import BaseParser

class YoutubeVideoParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Parses a YouTube URL, extracts metadata, downloads audio,
        transcribes audio using local faster-whisper model, and returns
        Paper, extracted links, and full transcription markdown.
        """
        url = source
        con.info(f"Extracting YouTube metadata and downloading audio for: {url}")
        
        # 1. Initialize metadata fallbacks
        video_id = ""
        video_title = None
        uploader = "Unknown Creator"
        description = ""
        duration = 0
        publish_date = ""
        
        # Extract ID from URL for stable ID generation
        id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if id_match:
            video_id = id_match.group(1)
            
        paper_id = f"yt_{video_id}" if video_id else f"yt_{slugify(url)}"
        
        temp_audio_path = None
        transcription_text = ""
        
        # 2. Ingest metadata and audio via yt-dlp
        try:
            import yt_dlp
            
            # Temporary directory to download the audio track
            temp_dir = tempfile.gettempdir()
            outtmpl = os.path.join(temp_dir, f"yt_{video_id if video_id else 'temp'}_%(id)s.%(ext)s")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': outtmpl,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                }],
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                con.dim("Fetching video details from YouTube...")
                info = ydl.extract_info(url, download=True)
                
                video_title = info.get("title", video_title)
                uploader = info.get("uploader", uploader)
                description = info.get("description", "")
                duration = info.get("duration", 0)
                publish_date = info.get("upload_date", "")
                
                # Retrieve actual audio file location (yt-dlp replaces extension post-conversion)
                expected_filename = ydl.prepare_filename(info)
                base, _ = os.path.splitext(expected_filename)
                temp_audio_path = f"{base}.m4a"
                
                con.success(f"Downloaded audio to {temp_audio_path}")
        except Exception as e:
            con.warning(f"Failed to download audio/metadata using yt-dlp: {e}. Falling back to metadata-only/description processing.")
            # Basic fallback title based on URL if yt-dlp failed completely
            if not video_title:
                video_title = f"YouTube Video ({url})"
                
        # 3. Transcribe audio using local faster-whisper model
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                from faster_whisper import WhisperModel
                
                model_path = "/Users/vladimirkasterin/models/audio/faster-whisper-large-v3-turbo"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Local Whisper model directory not found at {model_path}")
                
                con.info(f"Loading local Whisper model from {model_path} on CPU...")
                # large-v3-turbo on CPU must run with float32 or int8
                model = WhisperModel(model_path, device="cpu", compute_type="float32")
                
                con.info("Transcribing audio track...")
                segments, info_transcribe = model.transcribe(temp_audio_path, beam_size=5)
                
                transcript_segments = []
                for segment in segments:
                    # Format timestamp
                    h = int(segment.start // 3600)
                    m = int((segment.start % 3600) // 60)
                    s = int(segment.start % 60)
                    timestamp = f"[{h:02d}:{m:02d}:{s:02d}]" if h > 0 else f"[{m:02d}:{s:02d}]"
                    
                    transcript_segments.append(f"{timestamp} {segment.text.strip()}")
                    
                transcription_text = "\n".join(transcript_segments)
                con.success("Transcription completed successfully!")
            except Exception as e:
                con.warning(f"Local Whisper transcription failed: {e}. Using description as text body.")
                transcription_text = f"Transcription unavailable.\n\nDescription:\n{description}"
            finally:
                # Cleanup downloaded audio track
                try:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                        con.dim(f"Cleaned up temporary audio file: {temp_audio_path}")
                except Exception as cleanup_err:
                    con.warning(f"Could not clean up temp audio file: {cleanup_err}")
        else:
            transcription_text = f"Audio download failed.\n\nDescription:\n{description}"

        # 4. Extract references/links from description
        links = []
        url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
        for match in url_pattern.finditer(description):
            href = match.group(0).rstrip(".,;)-")
            if href not in links and href != url:
                links.append(href)

        # 5. Build Paper entity
        year_val = None
        if publish_date and len(publish_date) >= 4:
            try:
                year_val = int(publish_date[:4])
            except ValueError:
                pass
                
        video_title = video_title or "YouTube Video"
        paper = Paper(
            id=paper_id,
            title=video_title,
            authors=[uploader],
            year=year_val,
            abstract=description[:500] + ("..." if len(description) > 500 else ""),
            doi=None,
            file_path=url,
            properties={
                "source_type": "video",
                "url": url,
                "video_id": video_id,
                "uploader": uploader,
                "duration": duration,
                "publish_date": publish_date,
                "transcript": transcription_text
            }
        )
        
        # We append transcription text to the final full_text body so it gets indexed/chunked
        full_text_body = f"# {video_title}\n\nUploader: {uploader}\nLink: {url}\n\n## Transcript\n{transcription_text}\n\n## Description\n{description}"
        
        return paper, links, full_text_body
