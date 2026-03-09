#!/bin/bash

learning_rates=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)
temp_config="temp_config.json"
cp config/mobilenet_large.json "$temp_config"

best_lr=0
best_accuracy=0
best_loss=999999

echo "Starting grid search over learning rates..."

for lr in "${learning_rates[@]}"; do
    echo "Testing learning rate: $lr"
    
    # Update the config file with new learning rate
    jq --arg lr "$lr" '.learning_rate = ($lr | tonumber)' config/mobilenet_large.json > "$temp_config"
    mv "$temp_config" config/mobilenet_large.json
    
    # Run and save the output to a file
    torchrun --standalone --nproc_per_node=3 train.py config/mobilenet_large.json > output.txt 2>&1
    
    # Extract metrics more carefully
    if grep "Test validation run" output.txt > /dev/null; then
        accuracy=$(grep "Test validation run" output.txt | sed 's/.*Accuracy: \([0-9.]*\)%.*/\1/')
        loss=$(grep "Test validation run" output.txt | sed 's/.*Loss: \([0-9.]*\).*/\1/')
        
        echo "Learning Rate: $lr, Accuracy: $accuracy%, Loss: $loss"
        
        # Only compare if we got valid numbers
        if [ ! -z "$accuracy" ] && [ ! -z "$loss" ]; then
            if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                best_accuracy=$accuracy
                best_loss=$loss
                best_lr=$lr
            fi
        fi
    else
        echo "No validation metrics found in output for lr=$lr"
    fi
done

echo "Grid search complete!"
echo "Best results:"
echo "Learning Rate: $best_lr"
echo "Accuracy: $best_accuracy%"
echo "Loss: $best_loss"

# Cleanup
rm output.txt
cp mobilenet_large.json.bak mobilenet_large.json 2>/dev/null || true
