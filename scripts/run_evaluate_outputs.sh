MODEL_NAME="o4-mini"
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/qwen/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/qwen
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/qwen/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/qwen
