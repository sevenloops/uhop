from __future__ import annotations

import argparse
from pathlib import Path
from .metrics import write_kpi_snapshot, load_kpi_snapshot


def _fmt_table(rows, headers):
    # Simple column width computation (no external deps)
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, h in enumerate(headers):
            col_widths[i] = max(col_widths[i], len(str(r.get(h, ""))))
    def _line(vals):
        return " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(vals))
    sep = "-+-".join("-" * w for w in col_widths)
    out = [_line(headers), sep]
    for r in rows:
        out.append(_line([r.get(h, "") for h in headers]))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="UHOP KPI snapshot tool")
    ap.add_argument("--output", "-o", default=str(Path.home() / ".uhop_mvp_cache" / "metrics.json"), help="Output JSON path")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".uhop_mvp_cache"), help="Cache directory to read from")
    ap.add_argument("--show", action="store_true", help="Pretty-print snapshot after writing")
    args = ap.parse_args()
    outp = write_kpi_snapshot(args.output, cache_dir=args.cache_dir)
    print(f"Wrote KPI snapshot to {outp}")
    if args.show:
        snap = load_kpi_snapshot(outp)
        counts = snap.get("backend_selection_counts", {})
        print("\nBackend Selection Counts:")
        for b, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {b}: {c}")
        q = snap.get("backend_latency_quantiles", {})
        if q:
            print("\nBackend Latency Quantiles (from cache decisions):")
            rows = []
            for b, vals in sorted(q.items()):
                rows.append({
                    "backend": b,
                    "count": str(vals.get("count", 0)),
                    "p50": f"{vals.get('p50'):.3f}" if isinstance(vals.get('p50'), (int,float)) else "-",
                    "p90": f"{vals.get('p90'):.3f}" if isinstance(vals.get('p90'), (int,float)) else "-",
                    "p99": f"{vals.get('p99'):.3f}" if isinstance(vals.get('p99'), (int,float)) else "-",
                    "mean": f"{vals.get('mean'):.3f}" if isinstance(vals.get('mean'), (int,float)) else "-",
                })
            print(_fmt_table(rows, ["backend","count","p50","p90","p99","mean"]))
        ocl = snap.get("opencl_matmul", [])
        if ocl:
            print("\nOpenCL Matmul Perf (last run):")
            # filter rows with real numbers
            display_rows = []
            for r in ocl:
                display_rows.append({
                    "shape": r.get("shape"),
                    "kernel": r.get("kernel"),
                    "ms": f"{r.get('last_ms'):.4f}" if isinstance(r.get('last_ms'), (int,float)) else "-",
                    "gflops": f"{r.get('last_gflops'):.2f}" if isinstance(r.get('last_gflops'), (int,float)) else "-",
                })
            print(_fmt_table(display_rows, ["shape","kernel","ms","gflops"]))
        oclc = snap.get("opencl_conv2d", [])
        if oclc:
            print("\nOpenCL Conv2D Perf (stage timings):")
            display_rows = []
            for r in oclc:
                display_rows.append({
                    "shape": r.get("shape"),
                    "kernel": r.get("kernel"),
                    "ms": f"{r.get('last_ms'):.4f}" if isinstance(r.get('last_ms'), (int,float)) else "-",
                    "im2col": f"{r.get('im2col_ms'):.3f}" if isinstance(r.get('im2col_ms'), (int,float)) else "-",
                    "gemm": f"{r.get('gemm_ms'):.3f}" if isinstance(r.get('gemm_ms'), (int,float)) else "-",
                    "copy": f"{r.get('copy_ms'):.3f}" if isinstance(r.get('copy_ms'), (int,float)) else "-",
                    "chunk": str(r.get("chunked")),
                    "chunks": str(r.get("chunk_count") or "-"),
                    "var": f"{r.get('var_ms'):.3f}" if isinstance(r.get('var_ms'), (int,float)) else "-",
                    "retune": "yes" if r.get("retune_suggested") else "no",
                })
            print(_fmt_table(display_rows, ["shape","kernel","ms","im2col","gemm","copy","chunk","chunks","var","retune"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
