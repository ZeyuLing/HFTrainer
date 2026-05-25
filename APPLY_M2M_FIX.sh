#!/bin/bash

# Script to apply M2M text_guidance_scale fix to tools/infer.py
# This adds the missing --guidance-scale argument and passes it to the M2M pipeline

set -e

INFER_FILE="tools/infer.py"

if [ ! -f "$INFER_FILE" ]; then
    echo "ERROR: $INFER_FILE not found in current directory"
    exit 1
fi

echo "Backing up original file..."
cp "$INFER_FILE" "${INFER_FILE}.backup"

echo "Applying M2M text_guidance_scale fix..."

# Create a Python script to apply the changes
python3 << 'PYTHON_SCRIPT'
import re

infer_file = "tools/infer.py"

# Read the file
with open(infer_file, 'r') as f:
    content = f.read()

# Fix 1: Add --guidance-scale argument after --negative-prompt
negative_prompt_pattern = r"(    parser\.add_argument\('--negative-prompt', help='Negative prompt for motion/image generation\.'\))"
guidance_scale_arg = """    parser.add_argument('--guidance-scale', type=float, default=5.0,
                        help='CFG scale for text-conditioned models (default: 5.0)')"""

if "guidance-scale" not in content:
    content = re.sub(
        negative_prompt_pattern,
        r"\1\n" + guidance_scale_arg,
        content
    )
    print("✓ Added --guidance-scale CLI argument")
else:
    print("✓ --guidance-scale CLI argument already exists")

# Fix 2: Update M2M pipeline initialization to pass text_guidance_scale
m2m_pattern = r"(    pipeline = HyMotionM2MPipeline\(\n        bundle=bundle,\n        num_steps=args\.num_steps or 50,\n    \))"
m2m_replacement = r"""    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps or 50,
        text_guidance_scale=getattr(args, 'guidance_scale', 5.0) or 5.0,
    )"""

if "text_guidance_scale=getattr(args, 'guidance_scale'" not in content:
    content = re.sub(
        m2m_pattern,
        m2m_replacement,
        content
    )
    print("✓ Updated M2M pipeline to pass text_guidance_scale")
else:
    print("✓ M2M pipeline already passes text_guidance_scale")

# Write the fixed file
with open(infer_file, 'w') as f:
    f.write(content)

print("\n✓ File updated successfully!")
PYTHON_SCRIPT

echo ""
echo "Fix applied successfully!"
echo "Backup saved to: ${INFER_FILE}.backup"
echo ""
echo "Changes made:"
echo "  1. Added --guidance-scale CLI argument (default: 5.0)"
echo "  2. Updated M2M pipeline to pass text_guidance_scale parameter"
echo ""
echo "Verification:"
echo "  - Check that --guidance-scale is now available: python tools/infer.py --help"
echo "  - Test M2M inference: python tools/infer.py --config ... --checkpoint ... --input ... --output ... --guidance-scale 5.0"
