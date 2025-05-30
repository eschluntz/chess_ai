#!/usr/bin/env python
"""
Test timing for get_train_df with 1M samples
"""

from learning.eval_accuracy_helpers import get_train_df

if __name__ == "__main__":
    print("Testing get_train_df timing with 1M samples...")
    df = get_train_df(1_000_000, include_mates=False)
    print(f"Final result: {len(df)} samples")