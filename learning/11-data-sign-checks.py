#!/usr/bin/env python

"""
Check that evaluation functions return the correct signs for positions with extreme CP values.
Loads the first 1k positions from the Lichess dataset and tests the highest and lowest CP values.
"""

from datasets import load_dataset
import sys
import os
import chess

# Add parent directory to path to import eval functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import piece_value_eval, piece_position_eval


def main():
    """Check that evaluation functions return correct signs for extreme positions."""
    # Load the streaming dataset
    print("Loading Lichess chess position evaluations dataset (streaming)...")
    ds_full = load_dataset("Lichess/chess-position-evaluations", streaming=True)
    
    # Process the first 1000 positions
    num_samples = 1000
    print(f"\nProcessing first {num_samples} positions...")
    ds = ds_full["train"].take(num_samples)
    
    # Collect positions with cp values (not mate)
    positions = []
    mate_count = 0
    
    for example in ds:
        if example['cp'] is not None:
            positions.append({
                'fen': example['fen'],
                'cp': example['cp']
            })
        elif example['mate'] is not None:
            mate_count += 1
    
    print("\nData statistics:")
    print(f"  Total positions processed: {num_samples}")
    print(f"  Positions with CP values: {len(positions)}")
    print(f"  Positions with mate values: {mate_count}")
    
    if len(positions) == 0:
        print("No positions with CP values found!")
        return
    
    # Find positions with moderate CP values (5k-7k in absolute value)
    print("\nFinding positions with moderate CP values (5000-7000 in absolute value)...")
    
    positive_pos = None
    negative_pos = None
    
    for pos in positions:
        cp = pos['cp']
        abs_cp = abs(cp)
        
        # Find first position with CP between 5k and 7k
        if positive_pos is None and cp > 500 and cp < 7000:
            positive_pos = pos
            
        # Find first position with CP between -7k and -5k
        if negative_pos is None and cp < -500 and cp > -7000:
            negative_pos = pos
            
        # Stop if we found both
        if positive_pos is not None and negative_pos is not None:
            break
    
    if positive_pos is None:
        print("No position found with CP between 5000 and 7000")
        # Try to find any high positive value
        positive_pos = max(positions, key=lambda x: x['cp'])
        print(f"Using highest CP value instead: {positive_pos['cp']}")
    else:
        print(f"\nFound positive CP: {positive_pos['cp']} centipawns")
    print(f"FEN: {positive_pos['fen']}")
    
    if negative_pos is None:
        print("\nNo position found with CP between -7000 and -5000")
        # Try to find any low negative value
        negative_pos = min(positions, key=lambda x: x['cp'])
        print(f"Using lowest CP value instead: {negative_pos['cp']}")
    else:
        print(f"\nFound negative CP: {negative_pos['cp']} centipawns")
    print(f"FEN: {negative_pos['fen']}")
    
    # Test evaluation functions on these positions
    print("\n" + "="*50)
    print("Testing evaluation functions on extreme positions:")
    print("="*50)
    
    # Test positive CP position
    print(f"\n1. Position with positive CP ({positive_pos['cp']}):")
    board_positive = chess.Board(positive_pos['fen'])
    
    piece_value_score, _ = piece_value_eval(board_positive)
    piece_position_score, _ = piece_position_eval(board_positive)
    
    print(f"   piece_value_eval:    {piece_value_score:>6} (expected: positive)")
    print(f"   piece_position_eval: {piece_position_score:>6} (expected: positive)")
    
    # Check signs
    if piece_value_score <= 0:
        print("   ❌ WARNING: piece_value_eval returned non-positive score for positive CP position!")
    else:
        print("   ✓ piece_value_eval sign is correct")
        
    if piece_position_score <= 0:
        print("   ❌ WARNING: piece_position_eval returned non-positive score for positive CP position!")
    else:
        print("   ✓ piece_position_eval sign is correct")
    
    # Test negative CP position
    print(f"\n2. Position with negative CP ({negative_pos['cp']}):")
    board_negative = chess.Board(negative_pos['fen'])
    
    piece_value_score, _ = piece_value_eval(board_negative)
    piece_position_score, _ = piece_position_eval(board_negative)
    
    print(f"   piece_value_eval:    {piece_value_score:>6} (expected: negative)")
    print(f"   piece_position_eval: {piece_position_score:>6} (expected: negative)")
    
    # Check signs
    if piece_value_score >= 0:
        print("   ❌ WARNING: piece_value_eval returned non-negative score for negative CP position!")
    else:
        print("   ✓ piece_value_eval sign is correct")
        
    if piece_position_score >= 0:
        print("   ❌ WARNING: piece_position_eval returned non-negative score for negative CP position!")
    else:
        print("   ✓ piece_position_eval sign is correct")
    
    # Additional analysis: show piece counts
    print("\n" + "="*50)
    print("Piece count analysis:")
    print("="*50)
    
    for label, board, cp_value in [("Positive CP", board_positive, positive_pos['cp']), 
                                    ("Negative CP", board_negative, negative_pos['cp'])]:
        print(f"\n{label} position (CP: {cp_value}):")
        print("  White pieces:")
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count = len(board.pieces(piece_type, chess.WHITE))
            if count > 0:
                print(f"    {chess.piece_name(piece_type).capitalize()}: {count}")
        
        print("  Black pieces:")
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count = len(board.pieces(piece_type, chess.BLACK))
            if count > 0:
                print(f"    {chess.piece_name(piece_type).capitalize()}: {count}")
    
    print("\n" + "="*50)
    print("Sign check complete!")
    print("="*50)


if __name__ == "__main__":
    main()