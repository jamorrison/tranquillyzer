import pandas as pd
from itertools import product
from collections import defaultdict


def assign_cell_id(row, whitelist_df, barcode_columns):
    if len(barcode_columns) == 1:
        barcode_type = barcode_columns[0]
        corrected_sequences = row[f"corrected_{barcode_type}"].split(",")

        # Match against the whitelist
        matches = whitelist_df[whitelist_df[barcode_type].isin(corrected_sequences)]

        if len(matches) == 1:  # One exact match
            cell_id = matches.index[0] + 1  # 1-based indexing
            return cell_id
        else:  # Multiple matches or no match
            return "ambiguous"

    # Generalized multi-barcode matching based on the model-defined barcode columns.
    corrected_values = {}
    for barcode_type in barcode_columns:
        corrected_key = f"corrected_{barcode_type}"
        corrected_raw = row.get(corrected_key, "")
        corrected_values[barcode_type] = {
            token.strip() for token in str(corrected_raw).split(",") if token and token.strip() and token.strip() != "NMF"
        }

    combinations = list(product(*(sorted(corrected_values[barcode]) for barcode in barcode_columns)))
    matched_cells_by_count = defaultdict(set)

    for combination in combinations:
        mask = pd.Series(True, index=whitelist_df.index)
        for barcode_type, candidate in zip(barcode_columns, combination):
            mask &= whitelist_df[barcode_type] == candidate

        for idx in whitelist_df[mask].index:
            matched_cells_by_count[len(barcode_columns)].add(idx + 1)

        for match_count in range(len(barcode_columns) - 1, 0, -1):
            for matched_columns in product([True, False], repeat=len(barcode_columns)):
                if sum(matched_columns) != match_count:
                    continue
                partial_mask = pd.Series(True, index=whitelist_df.index)
                for include_col, barcode_type, candidate in zip(matched_columns, barcode_columns, combination):
                    if include_col:
                        partial_mask &= whitelist_df[barcode_type] == candidate
                for idx in whitelist_df[partial_mask].index:
                    matched_cells_by_count[match_count].add(idx + 1)

    best_match_count = max((count for count, cells in matched_cells_by_count.items() if cells), default=0)
    if best_match_count == 0:
        return "ambiguous"

    best_cells = matched_cells_by_count[best_match_count]
    if len(best_cells) != 1:
        return "ambiguous"

    return next(iter(best_cells))
