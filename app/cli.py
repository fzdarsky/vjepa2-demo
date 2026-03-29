import argparse
import json
import os
import sys
from pathlib import Path


from app.schemas import DEFAULT_NUM_FRAMES, DEFAULT_STRIDE, DEFAULT_TOP_K

DEFAULT_INPUT_DIR = "/input"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def cmd_serve(args):
    """Start FastAPI server via uvicorn."""
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port)


def _load_model():
    """Load the VJepa2Model. Separate function for test mocking."""
    from app.model import VJepa2Model, select_device
    model_path = os.environ.get("MODEL_PATH", "/model")
    if not Path(model_path).exists():
        print(
            f"No model found at {model_path}. "
            "Run `download` first or mount a model volume.",
            file=sys.stderr,
        )
        sys.exit(1)
    device = select_device(os.environ.get("DEVICE"))
    print(f"Loading model from {model_path}...", file=sys.stderr, end=" ", flush=True)
    model = VJepa2Model(model_path=model_path, device=device)
    print("done", file=sys.stderr)
    return model


def _find_videos(files: list[str]) -> list[Path]:
    """Resolve video file paths from args or scan /input/."""
    if files:
        return [Path(f) for f in files]
    input_dir = Path(DEFAULT_INPUT_DIR)
    if not input_dir.exists():
        print(f"No video files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    found = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not found:
        print(f"No video files found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    return found


def cmd_infer(args):
    """Run clip-based inference on video files."""
    from app.video import iter_clips
    from app.frames import save_clip_frames

    model = _load_model()
    videos = _find_videos(args.files)
    output_dir = Path(args.output)

    all_results = []

    for video_path in videos:
        print(f"Processing {video_path}...", file=sys.stderr)
        video_name = video_path.stem
        video_output = output_dir / video_name if args.save_frames else None

        clip_results = []
        for clip_index, clip in enumerate(iter_clips(video_path, args.num_frames, args.stride)):
            predictions = model.predict(clip.frames, top_k=args.top_k)
            is_partial = (clip.end_frame - clip.start_frame) < args.num_frames

            result = {
                "clip_index": clip_index,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "partial": is_partial,
                "predictions": [
                    {"label": p.label, "score": p.score} for p in predictions
                ],
            }
            clip_results.append(result)

            if args.format == "text":
                partial_marker = "  (partial)" if is_partial else ""
                print(
                    f"\nClip {clip_index} "
                    f"[frames {clip.start_frame}\u2013{clip.end_frame - 1}]"
                    f"{partial_marker}"
                )
                for p in predictions:
                    print(f"  {p.score:.3f}  {p.label}")
            elif args.format == "jsonl":
                print(json.dumps(result))

            if args.save_frames and video_output:
                save_clip_frames(
                    clip=clip,
                    clip_index=clip_index,
                    output_dir=video_output,
                    source_name=video_path.name,
                    predictions=predictions,
                    processor=model.processor,
                )

        all_results.extend(clip_results)

        if args.save_frames and video_output:
            results_path = video_output / "results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(clip_results, indent=2))
            print(f"Frames saved to {video_output}/", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(all_results, indent=2))

    print(f"{len(all_results)} clips processed", file=sys.stderr)


def cmd_download(args):
    """Download a HuggingFace model into a directory."""
    from huggingface_hub import snapshot_download
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} to {output_path}...", file=sys.stderr)
    snapshot_download(args.model, local_dir=output_path)
    print("Download complete.", file=sys.stderr)


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
    infer_parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    infer_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
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
