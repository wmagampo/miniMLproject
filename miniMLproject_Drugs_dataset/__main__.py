import pandas as pd
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def main():
    modes = [
        'generate-data',
        'train',
        'benchmark'
    ]

    parser.add_argument("--ratio", help="Ratio for training", type=float, default=0.8)
    parser.add_argument("--input-file", help="Input CSV file", type=str, required=True)
    parser.add_argument("--output-x-train-file", help="Output x train file", type=str, default="drugs_x_train.csv")
    parser.add_argument("--output-y-train-file", help="Output y train file", type=str, default="drugs_y_train.csv")
    # parser.add_argument("--output-x-test-file", help="Output x test file", type=str, default="x_test.csv")
    # parser.add_argument("--output-y-test-file", help="Output y test file", type=str, default="y_test.csv")
    # parser.add_argument("--mode", help="Type of action for MLTool", type=str, choices=modes, required=True)
    # parser.add_argument("--input-model-file", help="Input Model file", type=str, default="mlp.joblib")

    args = parser.parse_args()

    mode = args.mode
    ratio = args.ratio
    test_size = 1 - ratio
    input_file = args.input_file
    output_x_train_file = args.output_x_train_file
    output_y_train_file = args.output_y_train_file
    output_x_test_file = args.output_x_test_file
    output_y_test_file = args.output_y_test_file
    input_model_file = args.input_model_file


    if mode == 'generate-data':
        # Perform generate data
        cmd = GenerateData(params)
        cmd.run()

    elif mode == 'train':
        # Perform train mode (See: Example 2 notebook)
        cmd = Train()
        cmd.run()

    elif mode == 'benchmark':
        # Perform benchmark mode (See: Example 3 notebook)
        cmd = Benchmark()
        cmd.run()

    print("Done.")

if __name__ = "__main__":
    main()
