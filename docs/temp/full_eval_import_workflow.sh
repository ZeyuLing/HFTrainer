#!/bin/bash

# Complete end-to-end workflow for importing evaluation results into eval_dashboard
# Usage: bash full_eval_import_workflow.sh <eval_overfit_dir> <model_name>
#
# Example:
#   bash full_eval_import_workflow.sh \
#       work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
#       hymotion_m2m_v2_overfit_100

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <eval_overfit_dir> <model_name>"
    echo ""
    echo "Example:"
    echo "  $0 work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit hymotion_m2m_v2_overfit_100"
    exit 1
fi

EVAL_DIR="$1"
MODEL_NAME="$2"
IMPORT_DIR="${EVAL_DIR}/import_jsons_$(date +%Y%m%d_%H%M%S)"
DASHBOARD_DIR="motion_annot_web/eval_dashboard"

echo -e "${BLUE}=== Eval Dashboard Import Workflow ===${NC}"
echo -e "${BLUE}Evaluation Directory: ${EVAL_DIR}${NC}"
echo -e "${BLUE}Model Name: ${MODEL_NAME}${NC}"
echo ""

# Step 1: Verify eval directory
echo -e "${YELLOW}Step 1/5: Verifying evaluation directory...${NC}"
if [ ! -d "$EVAL_DIR" ]; then
    echo -e "${RED}✗ Directory not found: $EVAL_DIR${NC}"
    exit 1
fi

# Find summary.json files
SUMMARY_FILES=$(find "$EVAL_DIR" -maxdepth 2 -name "summary.json" -type f | sort)
NUM_MODES=$(echo "$SUMMARY_FILES" | grep -c "summary.json")

if [ $NUM_MODES -eq 0 ]; then
    echo -e "${RED}✗ No summary.json files found in $EVAL_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $NUM_MODES evaluation modes:${NC}"
while IFS= read -r summary_file; do
    mode_dir=$(dirname "$summary_file")
    mode_name=$(basename "$mode_dir")
    num_samples=$(find "$mode_dir" -maxdepth 1 -name "*.npz" -type f | wc -l)
    echo "  - $mode_name ($num_samples samples)"
done <<< "$SUMMARY_FILES"
echo ""

# Step 2: Create import directory
echo -e "${YELLOW}Step 2/5: Creating import directory...${NC}"
mkdir -p "$IMPORT_DIR"
echo -e "${GREEN}✓ Created: $IMPORT_DIR${NC}"
echo ""

# Step 3: Convert all summary.json files
echo -e "${YELLOW}Step 3/5: Converting summary.json to flat JSON format...${NC}"
NUM_CONVERTED=0
while IFS= read -r summary_file; do
    mode_dir=$(dirname "$summary_file")
    mode_name=$(basename "$mode_dir")
    
    # Run conversion
    if python3 prepare_eval_import.py \
        --summary-json "$summary_file" \
        --output-dir "$IMPORT_DIR" \
        --model-name "$MODEL_NAME" \
        --mode-dir "$mode_dir" \
        --setting "$mode_name" \
        2>/dev/null; then
        ((NUM_CONVERTED++))
    else
        echo -e "${RED}✗ Failed to convert $mode_name${NC}"
    fi
done <<< "$SUMMARY_FILES"

if [ $NUM_CONVERTED -eq 0 ]; then
    echo -e "${RED}✗ No modes were successfully converted${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Successfully converted $NUM_CONVERTED modes${NC}"
echo ""

# Step 4: Backup database
echo -e "${YELLOW}Step 4/5: Backing up database...${NC}"
if [ -f "$DASHBOARD_DIR/eval_dashboard.db" ]; then
    BACKUP_FILE="$DASHBOARD_DIR/eval_dashboard.db.bak_import_$(date +%Y%m%d_%H%M%S)"
    cp "$DASHBOARD_DIR/eval_dashboard.db" "$BACKUP_FILE"
    echo -e "${GREEN}✓ Backup created: $BACKUP_FILE${NC}"
else
    echo -e "${YELLOW}⚠ Database not found yet (first import)${NC}"
fi
echo ""

# Step 5: Import to database
echo -e "${YELLOW}Step 5/5: Importing to database...${NC}"
NUM_IMPORTED=0
for json_file in "$IMPORT_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        echo "  Importing: $(basename "$json_file")"
        if python3 "$DASHBOARD_DIR/data_importer.py" import "$json_file" \
            --notes "Batch import from $MODEL_NAME ($(date +%Y-%m-%d))" \
            2>/dev/null; then
            ((NUM_IMPORTED++))
        else
            echo -e "${RED}  ✗ Failed to import${NC}"
        fi
    fi
done

echo ""
echo -e "${GREEN}✓ Successfully imported $NUM_IMPORTED evaluation runs${NC}"
echo ""

# Summary
echo -e "${BLUE}=== Import Complete ===${NC}"
echo -e "${GREEN}✓ Model: $MODEL_NAME${NC}"
echo -e "${GREEN}✓ Import Directory: $IMPORT_DIR${NC}"
echo -e "${GREEN}✓ Evaluation Modes: $NUM_CONVERTED${NC}"
echo ""

echo -e "${BLUE}Next steps:${NC}"
echo "  1. Start the dashboard:"
echo "     cd $DASHBOARD_DIR && python3 app.py --port 8081"
echo ""
echo "  2. Open in browser:"
echo "     http://localhost:8081/task/E14"
echo ""
echo "  3. Verify the import:"
echo "     sqlite3 $DASHBOARD_DIR/eval_dashboard.db \\"
echo "         \"SELECT name, COUNT(*) FROM models m LEFT JOIN eval_runs r ON m.id=r.model_id GROUP BY m.id WHERE m.name='$MODEL_NAME';\""

