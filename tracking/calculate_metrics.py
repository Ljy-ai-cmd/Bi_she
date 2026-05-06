import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import print_results

def evaluate_tracker(tracker_name, tracker_param, dataset_name, sequence_name):
    # Load dataset
    dataset = get_dataset(dataset_name)
    # Filter for the specific sequence
    dataset = [s for s in dataset if s.name == sequence_name]
    
    if len(dataset) == 0:
        print(f"Sequence {sequence_name} not found in {dataset_name}")
        return

    # Initialize tracker
    tracker = Tracker(tracker_name, tracker_param, dataset_name)
    
    # Print results
    print(f"Results for {dataset_name}:")
    print_results([tracker], dataset, f'{dataset_name}_report', plot_types=('success', 'prec', 'norm_prec'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_name', type=str, default='sutrack')
    parser.add_argument('--tracker_param', type=str, default='sutrack_t224')
    parser.add_argument('--sequence', type=str, required=True)
    args = parser.parse_args()

    # Evaluate IR
    evaluate_tracker(args.tracker_name, args.tracker_param, 'antiuav_ir', args.sequence)
    
    # Evaluate RGB
    evaluate_tracker(args.tracker_name, args.tracker_param, 'antiuav_rgb', args.sequence)
