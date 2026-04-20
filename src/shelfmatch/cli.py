"""CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import rich.console
import rich.progress

from shelfmatch.pipeline.shelfmatcher import ShelfMatcher, PipelineConfig
from shelfmatch.training.synthetic import SyntheticShelfGenerator

console = rich.console.Console()


def cmd_detect(args):
    """Run detection on a shelf image."""
    matcher = ShelfMatcher(PipelineConfig(
        detector_type=args.detector,
        feature_model=args.feature,
        box_threshold=args.box_threshold,
    ))
    matcher.load_references(args.reference)
    results = matcher.detect(args.shelf)
    console.print(f"[green]Found {len(results.matches)} matches[/green]")
    for m in results.matches:
        console.print(f"  - {m.product_name} ({m.confidence}) sim={m.similarity:.3f}")


def cmd_synthesize(args):
    """Generate synthetic training data."""
    gen = SyntheticShelfGenerator(output_dir=args.output)

    for ref_path in args.reference:
        product_id = Path(ref_path).stem
        gen.add_product(product_id, ref_path)

    if args.background:
        for bg_path in args.background:
            gen.add_background(bg_path)
    else:
        gen.add_solid_background((180, 170, 160), (800, 600))

    gen.generate_dataset(
        num_images=args.num_images,
        min_products=args.min_products,
        max_products=args.max_products,
        output_dir=args.output,
    )
    console.print(f"[green]Generated {args.num_images} images → {args.output}[/green]")


def cmd_web(args):
    """Launch the web app."""
    from shelfmatch.webapp.main import main as web_main
    web_main()


def main():
    parser = argparse.ArgumentParser(prog="shelfmatch")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers()

    p_detect = sub.add_parser("detect", help="Detect products on a shelf image")
    p_detect.add_argument("-r", "--reference", nargs="+", required=True)
    p_detect.add_argument("-s", "--shelf", required=True)
    p_detect.add_argument("-d", "--detector", default="grounding_dino")
    p_detect.add_argument("-f", "--feature", default="siglip")
    p_detect.add_argument("--box-threshold", type=float, default=0.35)
    p_detect.set_defaults(fn=cmd_detect)

    p_synth = sub.add_parser("synthesize", help="Generate synthetic training data")
    p_synth.add_argument("-r", "--reference", nargs="+", required=True)
    p_synth.add_argument("-o", "--output", required=True)
    p_synth.add_argument("-n", "--num-images", type=int, default=100)
    p_synth.add_argument("--min-products", type=int, default=2)
    p_synth.add_argument("--max-products", type=int, default=8)
    p_synth.add_argument("-b", "--background", nargs="+")
    p_synth.set_defaults(fn=cmd_synthesize)

    p_web = sub.add_parser("web", help="Launch web UI")
    p_web.set_defaults(fn=cmd_web)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not hasattr(args, "fn"):
        parser.print_help()
        return

    args.fn(args)


if __name__ == "__main__":
    main()
