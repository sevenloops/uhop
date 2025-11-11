"""Minimal IR CLI for experimentation.

Usage examples:
  python -m uhop.cli_ir lower --file ir_matmul.json --out kernel.cl
  python -m uhop.cli_ir build --file ir_matmul.json
  python -m uhop.cli_ir validate --file ir_matmul.json --shape-set "A=64x128,B=128x32" --shape-set "A=96x64,B=64x16"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_ir(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inject_schedule(ir: dict, tile: int | None, vec: int | None) -> dict:
    if tile is None and vec is None:
        return ir
    d = json.loads(json.dumps(ir))  # deep copy
    sched = d.get("schedule") or {}
    if tile is not None:
        sched["tile_m"] = int(tile)
    if vec is not None:
        sched["vectorize"] = int(vec)
    d["schedule"] = sched
    return d


def cmd_lower(args):
    from .ir import ir_from_dict
    from .ir.opencl_lowering import lower_to_opencl

    raw = _load_ir(Path(args.file))
    ir = _inject_schedule(raw, args.tile, args.vec)
    op = ir_from_dict(ir)
    lowered = lower_to_opencl(op)
    out = Path(args.out or (Path(args.file).stem + ".cl"))
    out.write_text(lowered["source"], encoding="utf-8")
    print(json.dumps({"kernel_name": lowered.get("kernel_name"), "out": str(out), "tile": lowered.get("tile"), "vec": lowered.get("vec")}, indent=2))


def cmd_build(args):
    from .agent import _compile_kernel

    raw = _load_ir(Path(args.file))
    ir = _inject_schedule(raw, args.tile, args.vec)
    art = _compile_kernel({"ir": ir})
    print(json.dumps(art, indent=2))


def _parse_shape_set(s: str):
    # format: A=64x128,B=128x32
    parts = s.split(',')
    out = {}
    for p in parts:
        if '=' not in p:
            continue
        name, rhs = p.split('=', 1)
        dims = tuple(int(x) for x in rhs.split('x') if x)
        out[name.strip()] = list(dims)
    return out


def cmd_validate(args):
    from .agent import _validate

    raw = _load_ir(Path(args.file))
    ir = _inject_schedule(raw, args.tile, args.vec)
    shape_sets = [_parse_shape_set(s) for s in args.shape_set] if args.shape_set else None
    res = _validate({"ir": ir, "tolerance": args.tolerance, "shape_sets": shape_sets})
    print(json.dumps(res, indent=2))


def cmd_bench(args):
    # Compare vec=1 vs vec=4 (when N is aligned) for a single shape
    import numpy as np
    import pyopencl as cl  # type: ignore
    from .ir import ir_from_dict
    from .ir.opencl_lowering import lower_to_opencl

    raw = _load_ir(Path(args.file))
    # parse shapes from flags
    shap = _parse_shape_set(args.shape)
    A = tuple(int(x) for x in shap.get("A", []))
    B = tuple(int(x) for x in shap.get("B", []))
    if len(A) != 2 or len(B) != 2 or A[1] != B[0]:
        print("invalid --shape: need A=MxK,B=KxN", file=sys.stderr)
        sys.exit(2)
    M, K, N = A[0], A[1], B[1]

    def _time(vec):
        d = _inject_schedule(raw, args.tile, vec)
        op = ir_from_dict(d)
        low = lower_to_opencl(op)
        ctx = cl.create_some_context(interactive=False)
        q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        prg = cl.Program(ctx, low["source"]).build()
        kname = low.get("kernel_name")
        kern = getattr(prg, kname)
        rng = np.random.default_rng(0)
        hA = rng.random((M, K), dtype=np.float32)
        hB = rng.random((K, N), dtype=np.float32)
        hC = np.empty((M, N), dtype=np.float32)
        mf = cl.mem_flags
        dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hA)
        dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hB)
        dC = cl.Buffer(ctx, mf.WRITE_ONLY, size=hC.nbytes)
        ts = int(low.get("tile") or 16)
        gs0 = ((M + ts - 1) // ts) * ts
        gs1 = ((N + ts - 1) // ts) * ts
        kern.set_args(np.int32(M), np.int32(N), np.int32(K), dA, dB, dC)
        # warmups
        for _ in range(max(1, args.warmup)):
            e = cl.enqueue_nd_range_kernel(q, kern, (gs0, gs1), (ts, ts))
            e.wait()
        times = []
        for _ in range(max(1, args.iters)):
            e = cl.enqueue_nd_range_kernel(q, kern, (gs0, gs1), (ts, ts))
            e.wait()
            times.append((e.profile.end - e.profile.start) * 1e-6)
        return float(np.mean(times))

    t1 = _time(1)
    if N % 4 == 0:
        t4 = _time(4)
        print(json.dumps({"vec1_ms": t1, "vec4_ms": t4, "speedup": (t1 / t4 if t4 > 0 else None)}, indent=2))
    else:
        print(json.dumps({"vec1_ms": t1, "note": "N not divisible by 4; skipping vec=4"}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="UHOP IR helper CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_lower = sub.add_parser("lower", help="Lower IR JSON to kernel source")
    ap_lower.add_argument("--file", required=True)
    ap_lower.add_argument("--tile", type=int)
    ap_lower.add_argument("--vec", type=int)
    ap_lower.add_argument("--out")
    ap_lower.set_defaults(func=cmd_lower)

    ap_build = sub.add_parser("build", help="Lower + build kernel")
    ap_build.add_argument("--file", required=True)
    ap_build.add_argument("--tile", type=int)
    ap_build.add_argument("--vec", type=int)
    ap_build.set_defaults(func=cmd_build)

    ap_val = sub.add_parser("validate", help="Validate IR kernel correctness vs NumPy")
    ap_val.add_argument("--file", required=True)
    ap_val.add_argument("--tolerance", type=float, default=1e-4)
    ap_val.add_argument("--shape-set", action="append", help="Shape set like A=64x128,B=128x32 (repeatable)")
    ap_val.add_argument("--tile", type=int)
    ap_val.add_argument("--vec", type=int)
    ap_val.set_defaults(func=cmd_validate)

    ap_bench = sub.add_parser("bench", help="Benchmark vec=1 vs vec=4 when N divisible by 4")
    ap_bench.add_argument("--file", required=True)
    ap_bench.add_argument("--shape", required=True, help="A=MxK,B=KxN")
    ap_bench.add_argument("--tile", type=int)
    ap_bench.add_argument("--iters", type=int, default=5)
    ap_bench.add_argument("--warmup", type=int, default=2)
    ap_bench.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
