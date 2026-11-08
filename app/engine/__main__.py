"""python -m app.engine — run the perpetual engine without the Fragmenta UI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.engine", description="Fragmenta Perpetual Engine")
    ap.add_argument("--scene", default="places", help="scene file name in <data>/engine/scenes or a path")
    ap.add_argument("--layout", default="stereo", help="layout file name in <data>/engine/layouts or a path")
    ap.add_argument("--anchors", default=None, help="anchor file (default: same name as the scene, else places)")
    ap.add_argument("--program", default=None, help="program file name in <data>/engine/programs or a path (the piece: timed steps, looped)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("FRAGMENTA_ENGINE_PORT", "5002")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=9000, help="0 disables OSC input")
    ap.add_argument("--osc-out", default="127.0.0.1:9001", help="host:port for outbound OSC, '' disables")
    ap.add_argument("--renderer", choices=("torch", "fake"), default="torch")
    ap.add_argument("--sink", choices=("auto", "sounddevice", "file", "null", "radio"), default="auto")
    ap.add_argument("--radio", action="append", default=[], help="stream target for the radio sink, repeatable: icecast://source:PASSWORD@host:8000/tehran.mp3 (codec from the extension: .mp3 · .opus)")
    ap.add_argument("--device", default=None, help="PortAudio output device (index or name part) for the sounddevice sink")
    ap.add_argument("--channels", type=int, default=None, help="output channel count (default: the layout's)")
    ap.add_argument("--radio-name", default="Places I Have Always Never Been", help="stream name sent to Icecast")
    ap.add_argument("--clock", action="store_true", help="scenes follow the scene file's clock program (local time of the place)")
    ap.add_argument("--model", default="sa3-small-sfx-base")
    ap.add_argument("--lora", action="append", default=[], help="path[:strength], repeatable")
    ap.add_argument("--data", default=None, help="data folder (default: Fragmenta's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--play", action="store_true", help="start the transport at once")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-autoload", action="store_true", help="do not load the model at start")
    ap.add_argument("--fake-render-time", type=float, default=0.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("FRAGMENTA_LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s")
    log = logging.getLogger("engine")

    from app.core.config import get_config
    config = get_config()
    data_dir = Path(args.data) if args.data else config.user_data_dir
    loras = []
    for spec in args.lora:
        path, _, strength = spec.rpartition(":") if ":" in spec and not spec.endswith(".safetensors") else (spec, "", "")
        loras.append({"path": path or spec, "strength": float(strength) if strength else 1.0})

    from app.engine.api import create_app
    from app.engine.audio import radio_target
    from app.engine.engine import Engine
    engine = Engine(data_dir, scene=args.scene, layout=args.layout, anchors=args.anchors, renderer=args.renderer,
                    sink=args.sink, model_id=args.model, loras=loras, osc_port=args.osc_port, osc_out=args.osc_out or None,
                    seed=args.seed, fake_render_time=args.fake_render_time, config=config, program=args.program,
                    autoload=not args.no_autoload, resume=not args.no_resume,
                    radio=[radio_target(u, name=args.radio_name) for u in args.radio] or None,
                    clock=True if args.clock else None, device=args.device, channels=args.channels)
    engine.start()
    if args.play or getattr(engine, "_resume_playing", False):
        engine.transport(True)
    for n in engine.notes:
        log.info(n)

    stopping = threading.Event()

    def shutdown(*_a):
        if stopping.is_set():
            return
        stopping.set()
        log.info("stopping")
        engine.stop()
        # werkzeug has no clean stop from another thread; exit the process
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def watch_fatal() -> None:
        # a poisoned CUDA context cannot be repaired in-process: save, stop, and become a fresh
        # process with the same arguments (same PID, so a supervising backend keeps its handle)
        while not stopping.is_set():
            if getattr(engine, "fatal_exit", False):
                if os.environ.get("FRAGMENTA_ENGINE_NO_RESTART"):
                    log.error("fatal GPU error; self-restart disabled, exiting 75")
                    shutdown(); return
                stopping.set()
                log.error("fatal GPU error: %s — restarting the engine process in 5 s", engine.model_error)
                try:
                    engine.stop()
                except Exception:
                    pass
                threading.Event().wait(5.0)
                os.execv(sys.executable, [sys.executable, "-m", "app.engine", *sys.argv[1:]])
            threading.Event().wait(0.5)

    threading.Thread(target=watch_fatal, name="fatal-watch", daemon=True).start()
    app = create_app(engine, shutdown_cb=shutdown)
    log.info("engine %s on http://%s:%d/v1  OSC in :%s  data %s", engine.status()["version"], args.host, args.port,
             engine.osc_server.port if engine.osc_server else "off", engine.root)
    print(json.dumps({"ready": True, "port": args.port, "url": f"http://{args.host}:{args.port}/v1"}), flush=True)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False, debug=False)
    finally:
        shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
