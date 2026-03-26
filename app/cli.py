import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = "/input"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def cmd_serve(args):
    """Start FastAPI server via uvicorn."""
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port)


def cmd_infer(args):
    """Run clip-based inference on video files."""
    # Placeholder -- implemented in Task 7
    print("infer not yet implemented", file=sys.stderr)
    sys.exit(1)


def cmd_download(args):
    """Download a HuggingFace model into a directory."""
    # Placeholder -- implemented in Task 8
    print("download not yet implemented", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="vjepa2",
        description="V-JEPA2 Video Inference",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)

    # infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on video files")
    infer_parser.add_argument("files", nargs="*", help="Video file paths (default: scan /input/)")
    infer_parser.add_argument("--stride", type=int, default=None)
    infer_parser.add_argument("--num-frames", type=int, default=16)
    infer_parser.add_argument("--top-k", type=int, default=5)
    infer_parser.add_argument("--save-frames", action="store_true")
    infer_parser.add_argument("--output", default="/output")
    infer_parser.add_argument("--format", choices=["text", "json", "jsonl"], default="text")

    # download
    dl_parser = subparsers.add_parser("download", help="Download model from HuggingFace")
    dl_parser.add_argument("--model", required=True, help="HuggingFace model ID")
    dl_parser.add_argument("--output", default="/model")

    args = parser.parse_args()

    if args.command is None:
        args.command = "serve"
        args.host = "0.0.0.0"
        args.port = 8080

    commands = {
        "serve": cmd_serve,
        "infer": cmd_infer,
        "download": cmd_download,
    }
    commands[args.command](args)
