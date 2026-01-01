"""
Exploration script for the chess policy dataset.

Analyzes:
- Sample positions and their target moves
- Unique move vocabulary
- Move frequency distribution
- Move format patterns (promotions, standard moves)
"""

import chess
from collections import Counter

from data import get_raw_shuffled_data, extract_first_move


def explore_dataset():
    """Main exploration function."""
    print("=" * 60)
    print("Chess Policy Dataset Exploration")
    print("=" * 60)

    # Load full dataset
    print("\n1. Loading dataset...")
    df = get_raw_shuffled_data()
    print(f"   Total positions: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Show sample rows
    print("\n2. Sample positions:")
    print("-" * 60)
    for i, row in df.head(5).iterrows():
        first_move = extract_first_move(row["line"])
        print(f"   FEN: {row['fen'][:50]}...")
        print(f"   First move: {first_move}")
        print(f"   Full line: {row['line'][:60]}...")
        print(f"   CP: {row['cp']}, Mate: {row['mate']}")
        print()

    # Extract all first moves
    print("3. Extracting all first moves...")
    all_moves = df["line"].apply(extract_first_move).tolist()
    print(f"   Total moves extracted: {len(all_moves):,}")

    # Count unique moves
    print("\n4. Move vocabulary analysis:")
    print("-" * 60)
    move_counts = Counter(all_moves)
    unique_moves = len(move_counts)
    print(f"   Unique UCI moves: {unique_moves:,}")

    # Top moves
    print("\n   Top 20 most common moves:")
    for move, count in move_counts.most_common(20):
        pct = 100 * count / len(all_moves)
        print(f"      {move:8s} : {count:>8,} ({pct:5.2f}%)")

    # Move format analysis
    print("\n5. Move format analysis:")
    print("-" * 60)

    promotions = [m for m in move_counts.keys() if len(m) == 5]
    standard = [m for m in move_counts.keys() if len(m) == 4]

    print(f"   Standard moves (4 chars, e.g., e2e4): {len(standard):,}")
    print(f"   Promotions (5 chars, e.g., e7e8q): {len(promotions):,}")

    if promotions:
        print("\n   Sample promotions:")
        for m in sorted(promotions)[:10]:
            print(f"      {m} (count: {move_counts[m]:,})")

    # Analyze from/to squares
    print("\n6. From-square distribution:")
    print("-" * 60)
    from_squares = Counter(m[:2] for m in all_moves)
    print(f"   Unique from-squares used: {len(from_squares)}")
    print("   Top 10:")
    for sq, count in from_squares.most_common(10):
        pct = 100 * count / len(all_moves)
        print(f"      {sq}: {count:>8,} ({pct:5.2f}%)")

    print("\n7. To-square distribution:")
    print("-" * 60)
    to_squares = Counter(m[2:4] for m in all_moves)
    print(f"   Unique to-squares used: {len(to_squares)}")
    print("   Top 10:")
    for sq, count in to_squares.most_common(10):
        pct = 100 * count / len(all_moves)
        print(f"      {sq}: {count:>8,} ({pct:5.2f}%)")

    # Validate moves against chess rules
    print("\n8. Move validation (sample check):")
    print("-" * 60)
    valid_count = 0
    invalid_count = 0
    invalid_examples = []

    for i, row in df.head(1000).iterrows():
        fen = row["fen"]
        first_move = extract_first_move(row["line"])
        board = chess.Board(fen)

        uci_move = chess.Move.from_uci(first_move)
        if uci_move in board.legal_moves:
            valid_count += 1
        else:
            invalid_count += 1
            if len(invalid_examples) < 3:
                invalid_examples.append((fen, first_move))

    print(f"   Checked first 1000 positions:")
    print(f"   Valid moves: {valid_count}")
    print(f"   Invalid moves: {invalid_count}")

    if invalid_examples:
        print("   Invalid examples:")
        for fen, move in invalid_examples:
            print(f"      FEN: {fen}")
            print(f"      Move: {move}")

    # Summary stats for move encoding options
    print("\n" + "=" * 60)
    print("SUMMARY: Move Encoding Options")
    print("=" * 60)
    print(f"""
    Option 1: UCI Vocabulary
    - Vocabulary size: {unique_moves:,} unique moves
    - Pros: Direct mapping, no illegal move outputs
    - Cons: Larger vocabulary, must handle unseen moves at inference

    Option 2: From-To (64 x 64 = 4096)
    - Would need separate promotion head
    - Many positions never used (e.g., a1a8)
    - From-squares used: {len(from_squares)} / 64
    - To-squares used: {len(to_squares)} / 64

    Option 3: Factored (from:64 + to:64 + promo:5)
    - Smallest output dimension (133 total)
    - Requires 2-3 forward passes or multi-head
    - More complex training
    """)

    return move_counts


if __name__ == "__main__":
    move_counts = explore_dataset()
