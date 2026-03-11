"""
core/constraints/temporal_constraints.py
Advanced temporal and dependency constraint system
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Callable


class TemporalConstraintEngine:
    """
    Advanced temporal constraint engine
    """

    def __init__(self):
        self.temporal_constraints = {}
        self.dependency_graph = {}

    def add_temporal_constraint(self, name: str, constraint_type: str, params: Dict):
        """Add a temporal constraint"""

        constraint_handlers = {
            "SEQUENCE": self._create_sequence_constraint,
            "DURATION": self._create_duration_constraint,
            "INTERVAL": self._create_interval_constraint,
            "PERIODIC": self._create_periodic_constraint,
            "DEADLINE": self._create_deadline_constraint,
        }

        if constraint_type in constraint_handlers:
            constraint_func = constraint_handlers[constraint_type](params)
            self.temporal_constraints[name] = {
                "type": constraint_type,
                "params": params,
                "function": constraint_func,
            }

    def _create_sequence_constraint(self, params: Dict) -> Callable:
        """Create a sequence constraint (event A before B)"""

        event_a = params.get("event_a")
        event_b = params.get("event_b")

        return lambda events: (
            events[event_a]["timestamp"] < events[event_b]["timestamp"]
            if event_a in events and event_b in events
            else True
        )

    def _create_duration_constraint(self, params: Dict) -> Callable:
        """Create a duration constraint (duration between two events)"""

        event_a = params.get("event_a")
        event_b = params.get("event_b")
        min_duration = params.get("min_duration", 0)
        max_duration = params.get("max_duration", float("inf"))

        def duration_check(events):
            if event_a in events and event_b in events:
                duration = (
                    events[event_b]["timestamp"] - events[event_a]["timestamp"]
                ).total_seconds()
                return min_duration <= duration <= max_duration
            return True

        return duration_check

    def add_dependency_constraint(self, dependent: str, prerequisites: List[str]):
        """Add a dependency constraint"""

        if dependent not in self.dependency_graph:
            self.dependency_graph[dependent] = []

        self.dependency_graph[dependent].extend(prerequisites)

        # Create validation function
        def dependency_check(row):
            for prereq in prerequisites:
                if prereq not in row or not row[prereq]:
                    return False
            return True

        self.temporal_constraints[f"DEP_{dependent}"] = {
            "type": "DEPENDENCY",
            "dependent": dependent,
            "prerequisites": prerequisites,
            "function": dependency_check,
        }

    def validate_temporal_sequence(
        self, events_df: pd.DataFrame, sequence_col: str
    ) -> pd.DataFrame:
        """Validate temporal sequence"""

        events_df = events_df.copy()
        events_df["VALID_SEQUENCE"] = True

        # Verify that dates are in correct sequence
        if sequence_col in events_df.columns:
            for i in range(1, len(events_df)):
                if (
                    events_df.iloc[i][sequence_col]
                    < events_df.iloc[i - 1][sequence_col]
                ):
                    events_df.at[i, "VALID_SEQUENCE"] = False

        return events_df

    def generate_temporal_features(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> pd.DataFrame:
        """Generate additional temporal features"""

        df_temp = df.copy()

        if timestamp_col in df_temp.columns:
            # Convert to datetime if necessary
            if not pd.api.types.is_datetime64_any_dtype(df_temp[timestamp_col]):
                df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])

            # Extract temporal features
            df_temp["YEAR"] = df_temp[timestamp_col].dt.year
            df_temp["MONTH"] = df_temp[timestamp_col].dt.month
            df_temp["DAY"] = df_temp[timestamp_col].dt.day
            df_temp["HOUR"] = df_temp[timestamp_col].dt.hour
            df_temp["DAY_OF_WEEK"] = df_temp[timestamp_col].dt.dayofweek
            df_temp["IS_WEEKEND"] = df_temp["DAY_OF_WEEK"].isin([5, 6]).astype(int)

            # Time periods
            df_temp["QUARTER"] = df_temp[timestamp_col].dt.quarter
            df_temp["WEEK_OF_YEAR"] = df_temp[timestamp_col].dt.isocalendar().week

        return df_temp
