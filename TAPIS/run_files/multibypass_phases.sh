# Experiment setup
TRAIN_FOLD="fold2" # or fold1, train
TEST_FOLD="fold1" # or fold2
EXP_PREFIX="test_run" # costumize
TASK="PHASES"
ARCH="TAPIS"

#-------------------------
DATASET="MultiBypass"
DATASET_DIR="multibypasst40_challenge_trainval"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="configs/"$DATASET"/"$ARCH"/"$ARCH"_"$TASK".yaml"
OUTPUT_DIR="./outputs/"$DATASET"/"$TASK"/"$EXPERIMENT_NAME

# Change this variables if data is not located in ./data
FRAME_DIR="./data/"$DATASET_DIR"/videos_cutmargin512"
FRAME_LIST="./data/"$DATASET_DIR"/frame_lists"
ANNOT_DIR="./data/"$DATASET_DIR"/annotations/"
COCO_ANN_PATH="./data/"$DATASET_DIR"/annotations/multibypass_phases_"$TEST_FOLD".json"
# No pretrained MultiBypass checkpoint exists yet -- trains from scratch.
# GraSP's own PHASES.pyth is NOT a drop-in initialization here: NUM_CLASSES
# differs (11 GraSP phases vs 12 MultiBypass phases), so the final
# classifier layer shape would mismatch on a strict checkpoint load.

#-------------------------
# Run experiment

export PYTHONPATH=$(pwd)/tapis:$PYTHONPATH

mkdir -p $OUTPUT_DIR

python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TEST.ENABLE True \
TRAIN.ENABLE True \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "multibypass_phases_"$TRAIN_FOLD".json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON "multibypass_phases_"$TEST_FOLD".json" \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
TRAIN.BATCH_SIZE 24 \
TEST.BATCH_SIZE 24 \
OUTPUT_DIR $OUTPUT_DIR
