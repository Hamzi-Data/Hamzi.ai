"""
core/constraints/constraint_engine.py
Mathematical constraint engine - transforms linguistic constraints into mathematical masks
"""

import re
import json
import warnings
from typing import Dict, List, Callable, Any
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


class ConstraintParser:
    """
    Advanced constraint parser with support for common patterns (INEQUALITY, BETWEEN, MEMBERSHIP, LOGICAL, CONDITIONAL).
    Returns a unified analytical structure that is safe to compile and execute.
    """

    def __init__(self):
        self.constraint_types = {
            "INEQUALITY": "inequality",
            "EQUALITY": "equality",
            "MEMBERSHIP": "membership",
            "LOGICAL": "logical",
            "CONDITIONAL": "conditional",
            "TEMPORAL": "temporal",
            "GENERAL": "general",
        }

    def parse_constraint(self, constraint_str: str) -> Dict[str, Any]:
        """Parse constraint expression into a unified analytical structure"""
        if not isinstance(constraint_str, str):
            return {"error": "Constraint must be a string"}

        expr = constraint_str.strip()
        if expr == "":
            return {"error": "Empty constraint"}

        # IF ... THEN ...
        if re.search(r"\bIF\b.*\bTHEN\b", expr, re.IGNORECASE):
            parsed = self._parse_conditional(expr)
            if "error" not in parsed:
                return parsed

        # BETWEEN
        if re.search(r"\bBETWEEN\b", expr, re.IGNORECASE):
            parsed = self._parse_between(expr)
            if "error" not in parsed:
                return parsed

        # IN / NOT IN
        if re.search(r"\bIN\b|\bNOT IN\b", expr, re.IGNORECASE):
            parsed = self._parse_membership(expr)
            if "error" not in parsed:
                return parsed

        # AND/OR/NOT/XOR
        if re.search(r"\bAND\b|\bOR\b|\bNOT\b|\bXOR\b", expr, re.IGNORECASE):
            return self._parse_logical(expr)

        # Comparisons
        if any(op in expr for op in [">=", "<=", "==", "!=", ">", "<", "="]):
            parsed = self._parse_inequality(expr)
            if "error" not in parsed:
                return parsed

        # Unsupported
        return {"error": f"Unsupported constraint format: {expr}", "raw": expr}

    # -------------------------
    # Parsers
    # -------------------------
    def _parse_conditional(self, expr: str) -> Dict[str, Any]:
        """
        Pattern: IF <condition> THEN <consequence>
        """
        m = re.match(r"^\s*IF\s+(.+?)\s+THEN\s+(.+?)\s*$", expr, flags=re.IGNORECASE)
        if not m:
            return {"error": f"Invalid conditional syntax: {expr}", "raw": expr}
        cond_str = m.group(1).strip()
        then_str = m.group(2).strip()
        cond = self.parse_constraint(cond_str)
        then = self.parse_constraint(then_str)
        return {
            "type": "CONDITIONAL",
            "condition": cond,
            "consequence": then,
            "expression": expr,
        }

    def _parse_between(self, expr: str) -> Dict[str, Any]:
        """
        Pattern: column BETWEEN low AND high
        """
        pattern = r"^\s*(\w+)\s+BETWEEN\s+([+-]?\d+(?:\.\d+)?)\s+AND\s+([+-]?\d+(?:\.\d+)?)\s*$"
        m = re.match(pattern, expr, flags=re.IGNORECASE)
        if not m:
            return {"error": f"Invalid BETWEEN syntax: {expr}", "raw": expr}
        column = m.group(1)
        low = float(m.group(2))
        high = float(m.group(3))
        return {
            "type": "INEQUALITY",
            "operator": "BETWEEN",
            "column": column,
            "low": low,
            "high": high,
            "expression": expr,
        }

    def _parse_membership(self, expr: str) -> Dict[str, Any]:
        """
        Pattern: column IN (a, b, 'c') or column NOT IN (...)
        """
        pattern = r"^\s*(\w+)\s+(IN|NOT IN)\s*\(\s*(.+)\s*\)\s*$"
        m = re.match(pattern, expr, flags=re.IGNORECASE)
        if not m:
            return {"error": f"Invalid membership syntax: {expr}", "raw": expr}
        column = m.group(1)
        operator = m.group(2).upper()
        values_raw = m.group(3)
        parts = [v.strip() for v in re.split(r"\s*,\s*(?![^()]*\))", values_raw)]
        values = []
        for v in parts:
            if (v.startswith("'") and v.endswith("'")) or (
                v.startswith('"') and v.endswith('"')
            ):
                values.append(v[1:-1])
            else:
                try:
                    values.append(float(v) if "." in v else int(v))
                except Exception:
                    values.append(v)
        return {
            "type": "MEMBERSHIP",
            "operator": operator,
            "column": column,
            "values": values,
            "expression": expr,
        }

    def _parse_inequality(self, expr: str) -> Dict[str, Any]:
        """
        Pattern: column OP value or column OP other_column
        OP ∈ {>=, <=, >, <, ==, !=, =}
        """
        operators = [">=", "<=", "==", "!=", ">", "<", "="]
        for op in operators:
            if op in expr:
                parts = expr.split(op)
                if len(parts) != 2:
                    return {"error": f"Invalid inequality format: {expr}", "raw": expr}
                left = parts[0].strip()
                right = parts[1].strip()
                value = self._coerce_value(right)
                return {
                    "type": "INEQUALITY",
                    "operator": op,
                    "column": left,
                    "right": value,
                    "raw_right": right,
                    "expression": expr,
                }
        return {"error": f"No operator found in inequality: {expr}", "raw": expr}

    def _parse_logical(self, expr: str) -> Dict[str, Any]:
        """
        Parse simple compound logical expression (left→right).
        """
        expr = expr.strip()
        if expr.startswith("(") and expr.endswith(")"):
            # Remove outer parentheses if balanced
            depth = 0
            balanced = True
            for i, ch in enumerate(expr):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    balanced = False
                    break
            if balanced:
                expr = expr[1:-1].strip()

        tokens = re.split(
            r"(\bAND\b|\bOR\b|\bNOT\b|\bXOR\b)", expr, flags=re.IGNORECASE
        )
        sub_constraints = []
        logical_ops = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if re.fullmatch(r"AND|OR|NOT|XOR", token, flags=re.IGNORECASE):
                logical_ops.append(token.upper())
            else:
                sub_constraints.append(self.parse_constraint(token))
        return {
            "type": "LOGICAL",
            "operator": logical_ops if logical_ops else ["AND"],
            "sub_constraints": sub_constraints,
            "expression": expr,
        }

    # -------------------------
    # Utilities
    # -------------------------
    def _coerce_value(self, text: str) -> Any:
        """Convert text to appropriate type: number/date/text/column name"""
        t = text.strip()
        if (t.startswith("'") and t.endswith("'")) or (
            t.startswith('"') and t.endswith('"')
        ):
            return t[1:-1]
        # Special keywords
        if t.lower() in ("current_date", "today", "now"):
            return t.lower()
        # Number
        try:
            return float(t) if "." in t else int(t)
        except Exception:
            # Column name or text
            return t


class ConstraintEngine:
    """
    Main engine for managing and executing constraints
    """

    def __init__(self):
        self.parser = ConstraintParser()
        self.constraints: Dict[str, Dict[str, Any]] = {}
        self.constraint_masks: Dict[str, np.ndarray] = {}
        self.violation_tracker: Dict[str, List[Dict[str, Any]]] = {}

    # -------------------------
    # Constraint Management
    # -------------------------
    def add_constraint(self, name: str, constraint_str: str):
        parsed = self.parser.parse_constraint(constraint_str)
        if "error" in parsed:
            print(f"⚠️ Failed to parse constraint '{name}': {parsed.get('error')}")
            self.constraints[name] = {
                "raw": constraint_str,
                "parsed": parsed,
                "compiled": lambda row: None,  # Not applicable
                "violations": 0,
            }
            return

        compiled = self._safe_compile(parsed)
        self.constraints[name] = {
            "raw": constraint_str,
            "parsed": parsed,
            "compiled": compiled,
            "violations": 0,
        }
        print(f"✅ Constraint added: {name} -> {constraint_str}")

    def _safe_compile(
        self, parsed: Dict[str, Any]
    ) -> Callable[[Dict[str, Any]], bool | None]:
        """Safe compilation with missing column protection, returning None when not applicable."""
        ctype = parsed.get("type")
        try:
            if ctype == "INEQUALITY":
                func = self._compile_inequality(parsed)
                req = self._required_columns_inequality(parsed)
                return self._guard_missing_columns(func, req)
            elif ctype == "MEMBERSHIP":
                func = self._compile_membership(parsed)
                req = [parsed.get("column")]
                return self._guard_missing_columns(func, req)
            elif ctype == "LOGICAL":
                return self._compile_logical(parsed)
            elif ctype == "CONDITIONAL":
                return self._compile_conditional(parsed)
            else:
                return lambda row: None
        except Exception as e:
            print(f"⚠️ Error during constraint compilation: {e}")
            return lambda row: None

    def _guard_missing_columns(
        self, func: Callable[[Dict[str, Any]], bool | None], required_cols: List[str]
    ) -> Callable[[Dict[str, Any]], bool | None]:
        """Return None when required columns are missing instead of logging a violation."""
        required_cols = [c for c in required_cols if isinstance(c, str) and c]

        def wrapped(row: Dict[str, Any]) -> bool | None:
            for c in required_cols:
                if c not in row:
                    return None
            return func(row)

        return wrapped

    def _required_columns_inequality(self, parsed: Dict[str, Any]) -> List[str]:
        col = parsed.get("column")
        right = parsed.get("right")
        cols = [col]
        if (
            isinstance(right, str)
            and re.match(r"^\w+$", right)
            and right.lower() not in ("current_date", "today", "now")
        ):
            cols.append(right)
        return [c for c in cols if isinstance(c, str) and c]

    # -------------------------
    # Compile Constraints
    # -------------------------
    def _to_datetime_safe(self, v):
        if v is None or pd.isna(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return pd.Timestamp(v)
        try:
            return pd.to_datetime(v, errors="coerce")
        except Exception:
            return None

    def _compile_inequality(
        self, parsed: Dict[str, Any]
    ) -> Callable[[Dict[str, Any]], bool]:
        op = parsed.get("operator")
        col = parsed.get("column")

        # BETWEEN
        if op == "BETWEEN":
            low = parsed.get("low")
            high = parsed.get("high")

            def f_between(row):
                try:
                    val = row.get(col)
                    if pd.isna(val):
                        return False
                    return float(val) >= low and float(val) <= high
                except Exception:
                    return False

            return f_between

        right = parsed.get("right")

        # current_date / today / now
        if isinstance(right, str) and right.lower() in ("current_date", "today", "now"):
            right_dt = pd.Timestamp("now").normalize()

            def cmp_dt_const(row):
                val_dt = self._to_datetime_safe(row.get(col))
                if val_dt is None:
                    return False
                if op == "<=":
                    return val_dt <= right_dt
                if op == ">=":
                    return val_dt >= right_dt
                if op == "<":
                    return val_dt < right_dt
                if op == ">":
                    return val_dt > right_dt
                if op in ("==", "="):
                    return val_dt == right_dt
                if op == "!=":
                    return val_dt != right_dt
                return False

            return cmp_dt_const

        # Column-to-column comparison
        if isinstance(right, str) and re.match(r"^\w+$", right):
            other_col = right

            def cmp_cols(row):
                a = row.get(col)
                b = row.get(other_col)
                a_dt = self._to_datetime_safe(a)
                b_dt = self._to_datetime_safe(b)
                if a_dt is not None and b_dt is not None:
                    if op == "<=":
                        return a_dt <= b_dt
                    if op == ">=":
                        return a_dt >= b_dt
                    if op == "<":
                        return a_dt < b_dt
                    if op == ">":
                        return a_dt > b_dt
                    if op in ("==", "="):
                        return a_dt == b_dt
                    if op == "!=":
                        return a_dt != b_dt
                    return False
                # Numeric/text
                try:
                    af = float(a)
                    bf = float(b)
                    if op == "<=":
                        return af <= bf
                    if op == ">=":
                        return af >= bf
                    if op == "<":
                        return af < bf
                    if op == ">":
                        return af > bf
                    if op in ("==", "="):
                        return af == bf
                    if op == "!=":
                        return af != bf
                except Exception:
                    sa, sb = str(a), str(b)
                    if op in ("==", "="):
                        return sa == sb
                    if op == "!=":
                        return sa != sb
                    return False

            return cmp_cols

        # Fixed numeric/text value
        if isinstance(right, (int, float)):

            def cmp_numeric(row):
                try:
                    val = row.get(col)
                    if pd.isna(val):
                        return False
                    valf = float(val)
                    if op == ">=":
                        return valf >= right
                    if op == "<=":
                        return valf <= right
                    if op == ">":
                        return valf > right
                    if op == "<":
                        return valf < right
                    if op in ("==", "="):
                        return valf == right
                    if op == "!=":
                        return valf != right
                    return False
                except Exception:
                    return False

            return cmp_numeric
        else:

            def cmp_text(row):
                val = row.get(col)
                if pd.isna(val):
                    return False
                sval = str(val)
                if op in ("==", "="):
                    return sval == str(right)
                if op == "!=":
                    return sval != str(right)
                return False

            return cmp_text

    def _compile_membership(
        self, parsed: Dict[str, Any]
    ) -> Callable[[Dict[str, Any]], bool]:
        col = parsed.get("column")
        vals = parsed.get("values", [])
        op = parsed.get("operator", "IN")

        def f_in(row):
            try:
                v = row.get(col)
                # Flexible numeric/text comparison
                for candidate in vals:
                    try:
                        if isinstance(candidate, (int, float)):
                            if float(v) == float(candidate):
                                return op == "IN"
                        else:
                            if str(v) == str(candidate):
                                return op == "IN"
                    except Exception:
                        continue
                return op != "IN"
            except Exception:
                return False

        return f_in

    def _compile_logical(
        self, parsed: Dict[str, Any]
    ) -> Callable[[Dict[str, Any]], bool | None]:
        sub_parsed = parsed.get("sub_constraints", [])
        ops = parsed.get("operator", ["AND"])
        sub_funcs = [self._safe_compile(p) for p in sub_parsed]

        def f_logical(row):
            if not sub_funcs:
                return True
            res = sub_funcs[0](row)
            for i, op in enumerate(ops, start=1):
                if i >= len(sub_funcs):
                    break
                nxt = sub_funcs[i](row)
                # Propagate None (not applicable)
                if res is None or nxt is None:
                    res = None
                    continue
                if op == "AND":
                    res = bool(res) and bool(nxt)
                elif op == "OR":
                    res = bool(res) or bool(nxt)
                elif op == "XOR":
                    res = (bool(res) and not bool(nxt)) or (not bool(res) and bool(nxt))
                elif op == "NOT":
                    res = not bool(nxt)
            return res

        return f_logical

    def _compile_conditional(
        self, parsed: Dict[str, Any]
    ) -> Callable[[Dict[str, Any]], bool | None]:
        f_cond = self._safe_compile(parsed["condition"])
        f_then = self._safe_compile(parsed["consequence"])

        def f(row):
            c = f_cond(row)
            if c is None:
                return None  # Not applicable
            if c is True:
                t = f_then(row)
                return t
            else:
                return True  # Vacuous truth: if condition not met, constraint is considered satisfied

        return f

    # -------------------------
    # Validation and Application
    # -------------------------
    def validate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        for name, info in self.constraints.items():
            try:
                ok = info["compiled"](row)
                results[name] = ok
                if ok is False:
                    info["violations"] = info.get("violations", 0) + 1
                    self.violation_tracker.setdefault(name, []).append(row)
            except Exception as e:
                results[name] = None
                print(f"Error validating constraint {name}: {e}")
        return results

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results_df = df.copy()
        for name in self.constraints.keys():
            results_df[f"VALID_{name}"] = True

        for idx, row in results_df.iterrows():
            row_dict = row.to_dict()
            row_results = self.validate_row(row_dict)
            for name, ok in row_results.items():
                if ok is False:
                    results_df.at[idx, f"VALID_{name}"] = False
                elif ok is None:
                    results_df.at[idx, f"VALID_{name}"] = None
        return results_df

    def generate_constraint_mask(
        self, df: pd.DataFrame, constraint_name: str = None
    ) -> np.ndarray:
        masks = []
        names = [constraint_name] if constraint_name else list(self.constraints.keys())
        for name in names:
            if name not in self.constraints:
                continue
            func = self.constraints[name]["compiled"]
            mask = np.array(
                [
                    (
                        bool(func(row.to_dict()))
                        if func(row.to_dict()) is not None
                        else True
                    )
                    for _, row in df.iterrows()
                ],
                dtype=bool,
            )
            self.constraint_masks[name] = mask
            masks.append(mask)
        if not masks:
            return np.ones(len(df), dtype=bool)
        combined = np.logical_and.reduce(masks)
        return combined

    def enforce_constraints(
        self, generated_data: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """
        Apply simple constraints to generated data (numpy array).
        Supports BETWEEN and simple single-column comparisons.
        Does not automatically modify text/date constraints (can be extended later).
        """
        corrected = generated_data.copy()
        df = pd.DataFrame(corrected, columns=feature_names)

        for name, info in self.constraints.items():
            parsed = info.get("parsed", {})
            ctype = parsed.get("type")
            if ctype == "INEQUALITY":
                corrected = self._enforce_inequality(corrected, parsed, feature_names)
            elif ctype == "MEMBERSHIP":
                corrected = self._enforce_membership(corrected, parsed, feature_names)

        return corrected

    def _enforce_inequality(
        self, data: np.ndarray, parsed: Dict[str, Any], feature_names: List[str]
    ) -> np.ndarray:
        col = parsed.get("column")
        if col not in feature_names:
            return data
        idx = feature_names.index(col)
        op = parsed.get("operator")

        if op == "BETWEEN":
            low = parsed.get("low")
            high = parsed.get("high")
            data[:, idx] = np.clip(data[:, idx].astype(float), low, high)
            return data

        right = parsed.get("right")
        try:
            right_val = float(right)
            if op == ">=":
                data[:, idx] = np.maximum(data[:, idx].astype(float), right_val)
            elif op == "<=":
                data[:, idx] = np.minimum(data[:, idx].astype(float), right_val)
            elif op == ">":
                data[:, idx] = np.maximum(data[:, idx].astype(float), right_val + 1e-9)
            elif op == "<":
                data[:, idx] = np.minimum(data[:, idx].astype(float), right_val - 1e-9)
            elif op in ("==", "="):
                mask = np.isclose(data[:, idx].astype(float), right_val, atol=1e-6)
                data[~mask, idx] = right_val
            elif op == "!=":
                pass
        except Exception:
            pass

        return data

    def _enforce_membership(
        self, data: np.ndarray, parsed: Dict[str, Any], feature_names: List[str]
    ) -> np.ndarray:
        col = parsed.get("column")
        if col not in feature_names:
            return data
        # Do not automatically modify text values in this version; correction policies can be added later
        return data

    # -------------------------
    # Reports and Save/Load
    # -------------------------
    def generate_violation_report(self) -> str:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Constraint Violation Report")
        report_lines.append("=" * 80)
        total = 0
        for name, info in self.constraints.items():
            v = info.get("violations", 0)
            total += v
            report_lines.append(f"\nConstraint: {name}")
            report_lines.append(f"  Expression: {info.get('raw')}")
            report_lines.append(f"  Violation count: {v}")
            if v > 0 and name in self.violation_tracker:
                report_lines.append("  Example violations:")
                for i, row in enumerate(self.violation_tracker[name][:3]):
                    report_lines.append(f"    {i+1}. {row}")
        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"Total violations: {total}")
        return "\n".join(report_lines)

    def save_constraints(self, filepath: str):
        serializable = {}
        for name, info in self.constraints.items():
            serializable[name] = {
                "raw": info.get("raw"),
                "parsed": info.get("parsed"),
                "violations": info.get("violations", 0),
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    def load_constraints(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for name, info in loaded.items():
            self.add_constraint(name, info.get("raw", ""))
