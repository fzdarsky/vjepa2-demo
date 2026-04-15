#!/usr/bin/env python3
"""
Benchmark Report Generator

Reads stored benchmark results from benchmark/results/ and generates
a markdown summary report.

Usage:
    # Generate default report
    python -m benchmark.report

    # Filter by server/model
    python -m benchmark.report --server vjepa2-server-cuda
    python -m benchmark.report --model vit-l

    # Custom output path
    python -m benchmark.report --output benchmark/CUDA_RESULTS.md
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_results(results_dir: Path, server: str | None, model: str | None) -> list[dict]:
    """Load and filter benchmark results."""
    results = []
    for path in results_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                metadata = data.get("metadata", {})

                # Apply filters
                if server and metadata.get("server") != server:
                    continue
                if model and metadata.get("model") != model:
                    continue

                data["_path"] = path
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    # Sort by configuration (hardware, device, server, model, concurrency), then timestamp
    def sort_key(r):
        meta = r.get("metadata", {})
        return (
            meta.get("hardware", ""),
            meta.get("device", ""),
            meta.get("server", ""),
            meta.get("model", ""),
            meta.get("concurrency", 0),
            meta.get("timestamp", ""),
        )

    results.sort(key=sort_key)
    return results


def format_date(timestamp: str) -> str:
    """Format ISO timestamp to YYYY-MM-DD."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return timestamp[:10] if timestamp else "unknown"


def format_value(value, suffix: str = "", decimals: int = 2) -> str:
    """Format a numeric value with suffix."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def calculate_delta(current: float | None, previous: float | None) -> str:
    """Calculate percentage change from previous to current."""
    if current is None or previous is None or previous == 0:
        return "—"
    delta = ((current - previous) / previous) * 100
    if delta > 0:
        return f"+{delta:.0f}%"
    elif delta < 0:
        return f"{delta:.0f}%"
    return "0%"


def generate_summary_table(results: list[dict]) -> str:
    """Generate the main summary table."""
    lines = [
        "| Date | Hardware | Device | Server | Model | c | RTF | L_comp (ms) | Throughput | Version |",
        "| ---- | -------- | ------ | ------ | ----- | - | --- | ----------- | ---------- | ------- |",
    ]

    for r in results:
        meta = r.get("metadata", {})
        meth = r.get("methodology", {})
        summ = r.get("summary", {})
        env = meta.get("environment", {})

        date = format_date(meta.get("timestamp", ""))
        hardware = meta.get("hardware", "—")
        server = meta.get("server", "—")
        model = meta.get("model", "—")

        device = meta.get("device", "—")
        gpu = env.get("gpu")
        if gpu and device:
            device = f"{device} ({gpu})"

        concurrency = meta.get("concurrency", "—")
        rtf = format_value(meth.get("rtf"), decimals=2)
        l_comp = format_value(meth.get("L_comp_ms"), decimals=0)
        throughput = format_value(summ.get("throughput_rps"), " rps", decimals=2)
        # Use version if available, fall back to commit
        version = meta.get("version") or meta.get("commit", "—")
        if meta.get("dirty"):
            version += "*"

        lines.append(f"| {date} | {hardware} | {device} | {server} | {model} | {concurrency} | {rtf} | {l_comp} | {throughput} | {version} |")

    return "\n".join(lines)


def generate_server_sections(results: list[dict]) -> str:
    """Generate per-server breakdown sections."""
    # Group by server
    by_server = defaultdict(list)
    for r in results:
        server = r.get("metadata", {}).get("server", "unknown")
        by_server[server].append(r)

    sections = []
    for server, server_results in sorted(by_server.items()):
        lines = [
            f"### {server}",
            "",
            "| Date | Hardware | Device | Model | c | RTF | L_comp | Delta vs prev |",
            "| ---- | -------- | ------ | ----- | - | --- | ------ | ------------- |",
        ]

        # Build lookup of L_comp by full configuration for delta calculation
        # Configuration key = (hardware, device, model, concurrency)
        # Process oldest-to-newest to find "previous" result for each config
        sorted_oldest_first = sorted(
            server_results,
            key=lambda r: r.get("metadata", {}).get("timestamp", ""),
        )
        prev_by_config: dict[tuple, float] = {}
        delta_by_result: dict[int, str] = {}

        for r in sorted_oldest_first:
            meta = r.get("metadata", {})
            config_key = (
                meta.get("hardware", ""),
                meta.get("device", ""),
                meta.get("model", ""),
                meta.get("concurrency", 0),
            )
            l_comp = r.get("methodology", {}).get("L_comp_ms")

            prev_l_comp = prev_by_config.get(config_key)
            if prev_l_comp is not None and l_comp is not None:
                # Delta: how much faster is this vs previous run of same config?
                # Lower L_comp is better, so positive delta = improvement
                # e.g., prev=650, current=615 → (650-615)/650 = +5% improvement
                pct = ((prev_l_comp - l_comp) / prev_l_comp) * 100
                if pct > 0:
                    delta_by_result[id(r)] = f"+{pct:.0f}%"
                elif pct < 0:
                    delta_by_result[id(r)] = f"{pct:.0f}%"
                else:
                    delta_by_result[id(r)] = "0%"
            else:
                delta_by_result[id(r)] = "baseline"

            if l_comp is not None:
                prev_by_config[config_key] = l_comp

        # Render table in config order (hardware, device, model, concurrency, timestamp)
        sorted_results = sorted(
            server_results,
            key=lambda r: (
                r.get("metadata", {}).get("hardware", ""),
                r.get("metadata", {}).get("device", ""),
                r.get("metadata", {}).get("model", ""),
                r.get("metadata", {}).get("concurrency", 0),
                r.get("metadata", {}).get("timestamp", ""),
            ),
        )
        for r in sorted_results:
            meta = r.get("metadata", {})
            meth = r.get("methodology", {})

            date = format_date(meta.get("timestamp", ""))
            hardware = meta.get("hardware", "—")
            device = meta.get("device", "—")
            model = meta.get("model", "—")
            concurrency = meta.get("concurrency", "—")
            rtf = format_value(meth.get("rtf"), decimals=2)
            l_comp = meth.get("L_comp_ms")
            l_comp_str = format_value(l_comp, "ms", decimals=0)
            delta = delta_by_result.get(id(r), "—")

            lines.append(f"| {date} | {hardware} | {device} | {model} | {concurrency} | {rtf} | {l_comp_str} | {delta} |")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def generate_environment_details(results: list[dict]) -> str:
    """Generate collapsible environment details."""
    lines = []

    for r in results:
        meta = r.get("metadata", {})
        env = meta.get("environment", {})

        date = format_date(meta.get("timestamp", ""))
        hardware = meta.get("hardware", "unknown")
        device = meta.get("device", "unknown")
        server = meta.get("server", "unknown")
        model = meta.get("model", "unknown")
        concurrency = meta.get("concurrency", "?")

        summary_text = f"{date} {hardware}/{device} {server} {model} c={concurrency}"

        details = []
        if meta.get("video_file"):
            details.append(f"- Video: {meta['video_file']}")
        if meta.get("concurrency"):
            details.append(f"- Concurrency: {meta['concurrency']}")
        if meta.get("requests"):
            details.append(f"- Requests: {meta['requests']}")

        # Hardware info
        if env.get("instance_type"):
            details.append(f"- Instance: {env['instance_type']}")
        if env.get("cpu_model"):
            details.append(f"- CPU: {env['cpu_model']}")
        elif env.get("cpu"):  # backwards compat
            details.append(f"- CPU: {env['cpu']}")
        if env.get("cpu_count"):
            details.append(f"- CPU cores: {env['cpu_count']}")
        if env.get("memory_gb"):
            mem_str = f"{env['memory_gb']} GB"
            if env.get("memory_arch"):
                mem_str += f" ({env['memory_arch']})"
            details.append(f"- Memory: {mem_str}")

        # GPU info
        if env.get("gpu"):
            details.append(f"- GPU: {env['gpu']}")
        if env.get("cuda_version"):
            details.append(f"- CUDA: {env['cuda_version']}")
        if env.get("pytorch_version"):
            details.append(f"- PyTorch: {env['pytorch_version']}")
        if env.get("container_image"):
            details.append(f"- Container: {env['container_image']}")

        if details:
            lines.append(f"<details>")
            lines.append(f"<summary>{summary_text}</summary>")
            lines.append("")
            lines.extend(details)
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


def generate_report(results: list[dict]) -> str:
    """Generate the full markdown report."""
    today = datetime.now().strftime("%Y-%m-%d")

    sections = [
        "# Benchmark Results",
        "",
        f"Last updated: {today}",
        "",
        "## Summary",
        "",
        generate_summary_table(results),
        "",
        "## By Server",
        "",
        generate_server_sections(results),
        "",
        "## Environment Details",
        "",
        generate_environment_details(results),
    ]

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from stored results",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/results",
        help="Directory containing result JSON files (default: benchmark/results)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark/BENCHMARKS.md",
        help="Output markdown file (default: benchmark/BENCHMARKS.md)",
    )
    parser.add_argument(
        "--server",
        help="Filter results by server type",
    )
    parser.add_argument(
        "--model",
        help="Filter results by model",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run benchmarks first with: python -m benchmark.run ...")
        return

    results = load_results(results_dir, args.server, args.model)
    if not results:
        print("No benchmark results found")
        if args.server or args.model:
            print(f"  Filters: server={args.server}, model={args.model}")
        return

    print(f"Found {len(results)} benchmark result(s)")

    report = generate_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
