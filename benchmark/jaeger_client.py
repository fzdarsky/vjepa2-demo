"""
Jaeger Query API Client

Fetches traces from Jaeger and extracts span durations for benchmark analysis.
Supports time-window correlation for matching traces to benchmark requests.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass
class Span:
    """A single span from a trace."""

    trace_id: str
    span_id: str
    operation_name: str
    start_time_us: int  # microseconds since epoch
    duration_us: int  # microseconds
    tags: dict[str, Any] = field(default_factory=dict)
    parent_span_id: str | None = None

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0


@dataclass
class Trace:
    """A complete trace with all its spans."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @property
    def root_span(self) -> Span | None:
        """Return the root span (no parent)."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def duration_ms(self) -> float:
        """Total trace duration from root span."""
        root = self.root_span
        return root.duration_ms if root else 0.0


class JaegerClient:
    """Client for Jaeger Query API.

    Queries traces via Jaeger's HTTP API and extracts span timing data
    for benchmark analysis.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:16686",
        timeout: float = 30.0,
    ):
        """Initialize Jaeger client.

        Args:
            endpoint: Jaeger Query API base URL
            timeout: HTTP request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "JaegerClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def health_check(self) -> bool:
        """Check if Jaeger is reachable."""
        try:
            resp = self._client.get(f"{self.endpoint}/api/services")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    def get_services(self) -> list[str]:
        """List available services in Jaeger."""
        resp = self._client.get(f"{self.endpoint}/api/services")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def get_operations(self, service: str) -> list[str]:
        """List operations for a service."""
        resp = self._client.get(
            f"{self.endpoint}/api/services/{service}/operations"
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def get_traces(
        self,
        service: str,
        operation: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        min_duration: str | None = None,
        max_duration: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[Trace]:
        """Query Jaeger for traces.

        Args:
            service: Service name to query
            operation: Optional operation name filter
            start_time: Start of time window (default: 1 hour ago)
            end_time: End of time window (default: now)
            limit: Maximum number of traces to return
            min_duration: Minimum duration filter (e.g., "100ms", "1s")
            max_duration: Maximum duration filter
            tags: Tag filters as key-value pairs

        Returns:
            List of Trace objects with their spans
        """
        # Default time window: last hour
        now = datetime.now(timezone.utc)
        if end_time is None:
            end_time = now
        if start_time is None:
            start_time = datetime.fromtimestamp(
                end_time.timestamp() - 3600, tz=timezone.utc
            )

        # Convert to microseconds since epoch (Jaeger API format)
        start_us = int(start_time.timestamp() * 1_000_000)
        end_us = int(end_time.timestamp() * 1_000_000)

        params: dict[str, Any] = {
            "service": service,
            "start": start_us,
            "end": end_us,
            "limit": limit,
        }

        if operation:
            params["operation"] = operation
        if min_duration:
            params["minDuration"] = min_duration
        if max_duration:
            params["maxDuration"] = max_duration
        if tags:
            # Jaeger expects tags as JSON-encoded key:value pairs
            params["tags"] = str(tags)

        resp = self._client.get(f"{self.endpoint}/api/traces", params=params)
        resp.raise_for_status()
        data = resp.json()

        return self._parse_traces(data.get("data", []))

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a specific trace by ID."""
        resp = self._client.get(f"{self.endpoint}/api/traces/{trace_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        traces = self._parse_traces(data.get("data", []))
        return traces[0] if traces else None

    def _parse_traces(self, raw_traces: list[dict]) -> list[Trace]:
        """Parse Jaeger API response into Trace objects."""
        traces = []
        for raw_trace in raw_traces:
            trace_id = raw_trace.get("traceID", "")
            spans = []

            for raw_span in raw_trace.get("spans", []):
                # Extract tags into a dict
                tags = {}
                for tag in raw_span.get("tags", []):
                    tags[tag["key"]] = tag["value"]

                # Find parent span ID from references
                parent_id = None
                for ref in raw_span.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent_id = ref.get("spanID")
                        break

                span = Span(
                    trace_id=trace_id,
                    span_id=raw_span.get("spanID", ""),
                    operation_name=raw_span.get("operationName", ""),
                    start_time_us=raw_span.get("startTime", 0),
                    duration_us=raw_span.get("duration", 0),
                    tags=tags,
                    parent_span_id=parent_id,
                )
                spans.append(span)

            traces.append(Trace(trace_id=trace_id, spans=spans))

        return traces

    def extract_span_durations(
        self,
        traces: list[Trace],
        span_names: list[str],
    ) -> dict[str, list[float]]:
        """Extract durations for specific spans across traces.

        Args:
            traces: List of traces to analyze
            span_names: Span operation names to extract

        Returns:
            Dict mapping span name to list of durations in milliseconds
        """
        durations: dict[str, list[float]] = {name: [] for name in span_names}

        for trace in traces:
            for span in trace.spans:
                if span.operation_name in span_names:
                    durations[span.operation_name].append(span.duration_ms)

        return durations

    def wait_for_traces(
        self,
        service: str,
        start_time: datetime,
        expected_count: int,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
    ) -> list[Trace]:
        """Wait for traces to appear in Jaeger.

        Polls Jaeger until the expected number of traces are available
        or timeout is reached. Useful after sending requests to wait
        for trace data to flush.

        Args:
            service: Service name to query
            start_time: Only count traces after this time
            expected_count: Number of traces to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds

        Returns:
            List of traces found
        """
        deadline = time.monotonic() + timeout
        traces: list[Trace] = []

        while time.monotonic() < deadline:
            traces = self.get_traces(
                service=service,
                start_time=start_time,
                limit=expected_count * 2,  # Buffer for safety
            )

            if len(traces) >= expected_count:
                return traces[:expected_count]

            time.sleep(poll_interval)

        return traces  # Return what we have, even if incomplete
