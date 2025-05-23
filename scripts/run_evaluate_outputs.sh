MODEL_NAME="o4-mini"
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/chatgpt/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/chatgpt
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/mistral/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/mistral
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/qwen/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/qwen
python evaluate_outputs.py -m $MODEL_NAME -i ./generated_scripts/llama/generated_scripts.csv -o ./evaluation_results/$MODEL_NAME/llama

