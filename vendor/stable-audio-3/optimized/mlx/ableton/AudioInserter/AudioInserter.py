import Live
import socket
import threading
import os
import json
import re

from _Framework.ControlSurface import ControlSurface

SOCKET_HOST = "127.0.0.1"
SOCKET_PORT = 9129
BUFFER_SIZE  = 4096


class AudioInserter(ControlSurface):

    def __init__(self, c_instance):
        super().__init__(c_instance)
        self._c_instance = c_instance
        self._server_thread = None
        self._running = False
        self._start_server()
        self.log_message("AudioInserter: started, listening on port %d" % SOCKET_PORT)

    def _start_server(self):
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

    def _server_loop(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((SOCKET_HOST, SOCKET_PORT))
            srv.listen(5)
            srv.settimeout(1.0)
            while self._running:
                try:
                    conn, _ = srv.accept()
                    data = conn.recv(BUFFER_SIZE).decode("utf-8").strip()
                    response = self._handle_command(data)
                    conn.sendall((response + "\n").encode("utf-8"))
                    conn.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    self.log_message("AudioInserter socket error: %s" % str(e))
        except Exception as e:
            self.log_message("AudioInserter server error: %s" % str(e))

    def _handle_command(self, raw):
        try:
            cmd = json.loads(raw)
        except Exception:
            return json.dumps({"ok": False, "error": "Invalid JSON"})

        action = cmd.get("action", "")

        if action == "ping":
            return json.dumps({"ok": True, "message": "pong"})

        if action == "insert_audio":
            file_path = cmd.get("file_path", "")
            if not file_path:
                return json.dumps({"ok": False, "error": "No file_path provided"})
            if not os.path.isfile(file_path):
                return json.dumps({"ok": False, "error": "File not found: %s" % file_path})

            result = {"done": False, "ok": False, "error": None}
            event = threading.Event()
            track_name = cmd.get("track_name", "")

            def do_insert():
                try:
                    result["ok"], result["error"] = self._insert_audio(file_path, track_name)
                except Exception as e:
                    result["ok"] = False
                    result["error"] = str(e)
                finally:
                    result["done"] = True
                    event.set()

            self.schedule_message(1, do_insert)
            event.wait(timeout=10.0)

            if result["ok"]:
                msg = result.get("error") or ("Inserted: %s" % os.path.basename(file_path))
                return json.dumps({"ok": True, "message": msg})
            else:
                return json.dumps({"ok": False, "error": result.get("error", "Unknown error")})

        return json.dumps({"ok": False, "error": "Unknown action: %s" % action})

    def _clean_track_name(self, name):
        name = re.sub(r'\b\w+\s*:\s*\S+', '', name)
        name = re.sub(r'\s{2,}', ' ', name).strip()
        return name if name else "Audio"

    def _insert_audio(self, file_path, track_name=""):
        import shutil, time
        song = self._c_instance.song()
        insert_time = song.current_song_time

        song.create_audio_track(-1)
        track = song.tracks[-1]

        raw_name = track_name if track_name else os.path.splitext(os.path.basename(file_path))[0]
        clean_name = self._clean_track_name(raw_name)
        track.name = clean_name

        # Copy file to a unique path so each generation is preserved independently
        ext = os.path.splitext(file_path)[1]
        unique_name = "%s_%s%s" % (clean_name.replace(" ", "_"), int(time.time()), ext)
        dest_dir = os.path.dirname(file_path)
        unique_path = os.path.join(dest_dir, unique_name)
        shutil.copy2(file_path, unique_path)
        self.log_message("AudioInserter: copied to %s" % unique_path)

        # Step 1: create audio clip in session slot 0
        slot = track.clip_slots[0]
        slot.create_audio_clip(unique_path)
        clip = slot.clip

        if clip is None:
            return False, "clip is None after create_audio_clip"

        clip.name = clean_name

        self.log_message("AudioInserter: session clip created, length=%s is_session_clip=%s" % (
            clip.length, clip.is_session_clip))

        # Step 2: duplicate to arrangement at playhead using track.duplicate_clip_to_arrangement
        # Signature from Live 11 sources: duplicate_clip_to_arrangement(clip, position)
        try:
            track.duplicate_clip_to_arrangement(clip, insert_time)
            self.log_message("AudioInserter: duplicate_clip_to_arrangement succeeded")

            # Step 3: delete the session clip — we only want it in arrangement
            slot.delete_clip()

            return True, None
        except Exception as e:
            self.log_message("AudioInserter: duplicate_clip_to_arrangement failed: %s" % str(e))
            return False, "duplicate_clip_to_arrangement failed: %s" % str(e)

    def disconnect(self):
        self._running = False
        super().disconnect()
        self.log_message("AudioInserter: disconnected")