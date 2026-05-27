import random

class DatasetGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)

    def generate_tasks(self):
        """
        Generates 20 tasks.
        Tasks 0-9: Arithmetic sequences (Rule A)
        Tasks 10-19: Geometric sequences (Rule B - Paradigm Shift)
        
        Applies 30% occlusion by masking 1 of the 4 context elements in each sequence with '?'.
        """
        tasks = []
        
        # Rule A: Arithmetic (Tasks 0-9)
        arithmetic_configs = [
            (3, 4),    # starts at 3, diff 4  -> 3, 7, 11, 15 -> next 19
            (5, 3),    # starts at 5, diff 3  -> 5, 8, 11, 14 -> next 17
            (10, 10),  # starts at 10, diff 10 -> 10, 20, 30, 40 -> next 50
            (1, 6),    # starts at 1, diff 6  -> 1, 7, 13, 19 -> next 25
            (20, -3),  # starts at 20, diff -3 -> 20, 17, 14, 11 -> next 8
            (100, -10),# starts at 100, diff -10 -> 100, 90, 80, 70 -> next 60
            (2, 7),    # starts at 2, diff 7  -> 2, 9, 16, 23 -> next 30
            (25, -4),  # starts at 25, diff -4 -> 25, 21, 17, 13 -> next 9
            (7, 8),    # starts at 7, diff 8  -> 7, 15, 23, 31 -> next 39
            (18, -3),  # starts at 18, diff -3 -> 18, 15, 12, 9 -> next 6
        ]

        for idx, (start, diff) in enumerate(arithmetic_configs):
            seq = [start + i * diff for i in range(5)] # 4 context, 1 target
            context = seq[:4]
            target = seq[4]
            
            # Apply 30% occlusion (mask 1 element of the 4 context elements, avoiding the first element)
            masked_idx = random.choice([1, 2, 3])
            occluded = list(context)
            occluded[masked_idx] = "?"
            
            prompt = f"Identify the logical pattern in the sequence and predict the next number. Sequence: {', '.join(str(x) for x in occluded)}, ?. Next number is:"
            
            tasks.append({
                "id": idx,
                "type": "arithmetic",
                "raw_sequence": seq,
                "occluded_sequence": occluded,
                "prompt": prompt,
                "expected": str(target)
            })

        # Rule B: Geometric (Tasks 10-19 - Paradigm Shift)
        geometric_configs = [
            (2, 2),    # 2, 4, 8, 16 -> next 32
            (3, 3),    # 3, 9, 27, 81 -> next 243
            (1, 5),    # 1, 5, 25, 125 -> next 625
            (10, 2),   # 10, 20, 40, 80 -> next 160
            (2, 3),    # 2, 6, 18, 54 -> next 162
            (64, 0.5), # 64, 32, 16, 8 -> next 4
            (243, 0.3333333333333333), # 243, 81, 27, 9 -> next 3 (convert to int)
            (1, 4),    # 1, 4, 16, 64 -> next 256
            (5, 2),    # 5, 10, 20, 40 -> next 80
            (3, 2),    # 3, 6, 12, 24 -> next 48
        ]

        for idx, (start, ratio) in enumerate(geometric_configs):
            seq = [int(start * (ratio ** i)) for i in range(5)]
            context = seq[:4]
            target = seq[4]
            
            # Apply 30% occlusion
            masked_idx = random.choice([1, 2, 3])
            occluded = list(context)
            occluded[masked_idx] = "?"
            
            prompt = f"Identify the logical pattern in the sequence and predict the next number. Sequence: {', '.join(str(x) for x in occluded)}, ?. Next number is:"
            
            tasks.append({
                "id": idx + 10,
                "type": "geometric",
                "raw_sequence": seq,
                "occluded_sequence": occluded,
                "prompt": prompt,
                "expected": str(target)
            })

        return tasks
