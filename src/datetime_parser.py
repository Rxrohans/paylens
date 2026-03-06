"""
DateTime Parser for PayLens
Converts temporal references in queries to concrete dates for accurate web searches.
V3 feature 1
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DateTimeParser:
    """Parse temporal references and convert them to concrete dates."""
    
    # Temporal patterns to detect (order matters for multi-word phrases)
    TEMPORAL_PATTERNS = {
        # Multi-word patterns first (to avoid partial matches)
        'last 30 days': (-30, 'days'),
        'last 7 days': (-7, 'days'),
        'last 3 days': (-3, 'days'),
        'last 48 hours': (-2, 'days'),
        'last 24 hours': (-1, 'days'),
        'last week': (-7, 'days'),
        'last month': (-1, 'months'),
        'last year': (-1, 'years'),
        'this week': (0, 'weeks'),
        'this month': (0, 'months'),
        'this year': (0, 'years'),
        
        # Single word patterns
        'today': (0, 'days'),
        'yesterday': (-1, 'days'),
        'tomorrow': (1, 'days'),
        
        # Weekday references
        'monday': ('weekday', 0),
        'tuesday': ('weekday', 1),
        'wednesday': ('weekday', 2),
        'thursday': ('weekday', 3),
        'friday': ('weekday', 4),
        'saturday': ('weekday', 5),
        'sunday': ('weekday', 6),
        
        # Recent time periods
        'recent': (-3, 'days'),
        'recently': (-3, 'days'),
        'latest': (0, 'days'),
    }
    
    def __init__(self):
        self.today = datetime.now()
        logger.info(f"DateTimeParser initialized with current date: {self.today.strftime('%Y-%m-%d')}")
    
    def detect_temporal_references(self, query: str) -> List[str]:
        """
        Detect temporal references in query.
        
        Args:
            query: User query string
            
        Returns:
            List of detected temporal keywords (sorted by position in query)
        """
        query_lower = query.lower()
        detected = []
        
        # Sort patterns by length (longest first) to match multi-word phrases first
        sorted_patterns = sorted(
            self.TEMPORAL_PATTERNS.keys(),
            key=len,
            reverse=True
        )
        
        for pattern in sorted_patterns:
            # Use word boundaries to avoid partial matches
            # E.g., "today" should not match "yesterday"
            pattern_regex = r'\b' + re.escape(pattern) + r'\b'
            if re.search(pattern_regex, query_lower):
                if pattern not in detected:  # Avoid duplicates
                    detected.append(pattern)
        
        if detected:
            logger.info(f"Detected temporal references: {detected}")
        
        return detected
    
    def resolve_to_date(self, temporal_ref: str) -> Optional[str]:
        """
        Convert temporal reference to concrete date string.
        
        Args:
            temporal_ref: Temporal keyword (e.g., 'yesterday', 'last week')
            
        Returns:
            Date string in 'YYYY-MM-DD' format or None if not recognized
        """
        temporal_ref_lower = temporal_ref.lower()
        
        if temporal_ref_lower not in self.TEMPORAL_PATTERNS:
            logger.warning(f"Unknown temporal reference: {temporal_ref}")
            return None
        
        pattern_info = self.TEMPORAL_PATTERNS[temporal_ref_lower]
        
        # Handle weekday references
        if isinstance(pattern_info, tuple) and pattern_info[0] == 'weekday':
            target_weekday = pattern_info[1]
            current_weekday = self.today.weekday()
            
            # Calculate days back to last occurrence
            days_back = (current_weekday - target_weekday) % 7
            if days_back == 0:
                days_back = 7  # Go to last week if today is that day
            
            target_date = self.today - timedelta(days=days_back)
            date_str = target_date.strftime('%Y-%m-%d')
            logger.debug(f"Resolved '{temporal_ref}' to {date_str} (weekday reference)")
            return date_str
        
        # Handle relative time periods
        delta_value, delta_unit = pattern_info
        
        if delta_unit == 'days':
            target_date = self.today + timedelta(days=delta_value)
        elif delta_unit == 'weeks':
            target_date = self.today + timedelta(weeks=delta_value)
        elif delta_unit == 'months':
            # Approximate month as 30 days
            target_date = self.today + timedelta(days=delta_value * 30)
        elif delta_unit == 'years':
            # Approximate year as 365 days
            target_date = self.today + timedelta(days=delta_value * 365)
        else:
            logger.warning(f"Unknown delta unit: {delta_unit}")
            return None
        
        date_str = target_date.strftime('%Y-%m-%d')
        logger.debug(f"Resolved '{temporal_ref}' to {date_str} ({delta_value} {delta_unit})")
        return date_str
    
    def get_date_range(self, temporal_ref: str) -> Optional[Tuple[str, str]]:
        """
        Get date range for temporal reference.
        Useful for queries that need a time window.
        
        Args:
            temporal_ref: Temporal keyword
            
        Returns:
            Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
        """
        temporal_ref_lower = temporal_ref.lower()
        
        if temporal_ref_lower == 'today':
            date_str = self.today.strftime('%Y-%m-%d')
            return (date_str, date_str)
        
        if temporal_ref_lower == 'this week':
            # Start of week (Monday)
            start = self.today - timedelta(days=self.today.weekday())
            # End of week (Sunday)
            end = start + timedelta(days=6)
            return (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        if temporal_ref_lower == 'this month':
            # Start of month
            start = self.today.replace(day=1)
            # Last day of month
            if self.today.month == 12:
                end = self.today.replace(day=31)
            else:
                end = (self.today.replace(month=self.today.month + 1, day=1) 
                       - timedelta(days=1))
            return (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        if temporal_ref_lower == 'this year':
            start = self.today.replace(month=1, day=1)
            end = self.today.replace(month=12, day=31)
            return (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        # For other references, use single date
        date_str = self.resolve_to_date(temporal_ref_lower)
        if date_str:
            return (date_str, date_str)
        
        return None
    
    def augment_query(self, query: str) -> str:
        """
        Augment query with concrete date context for web search.
        
        Example:
            "What happened today with UPI?" 
            → "What happened today with UPI? [today = 2026-03-06]"
        
        Args:
            query: Original user query
            
        Returns:
            Augmented query with date context appended
        """
        detected = self.detect_temporal_references(query)
        
        if not detected:
            return query
        
        # Build date context
        date_contexts = []
        for ref in detected:
            date_str = self.resolve_to_date(ref)
            if date_str:
                date_contexts.append(f"{ref} = {date_str}")
        
        if date_contexts:
            context_str = " [" + ", ".join(date_contexts) + "]"
            augmented = query + context_str
            logger.info(f"Augmented query: '{query}' → '{augmented}'")
            return augmented
        
        return query
    
    def get_current_date_context(self) -> str:
        """
        Get current date context for LLM prompt.
        
        Returns:
            String with current date info formatted for LLM
        """
        weekday = self.today.strftime('%A')
        month = self.today.strftime('%B')
        day = self.today.day
        year = self.today.year
        
        # Add ordinal suffix (1st, 2nd, 3rd, 4th, etc.)
        if 10 <= day <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        
        return f"Current date: {self.today.strftime('%Y-%m-%d')} ({weekday}, {month} {day}{suffix}, {year})"
    
    def is_temporal_query(self, query: str) -> bool:
        """
        Quick check if query contains temporal references.
        
        Args:
            query: User query string
            
        Returns:
            Boolean indicating if temporal references found
        """
        return len(self.detect_temporal_references(query)) > 0


def parse_datetime_query(query: str) -> Dict[str, any]:
    """
    Utility function to parse query and extract datetime info.
    
    This is the main entry point for other modules.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary with:
            - original_query: Original query
            - augmented_query: Query with date context for web search
            - temporal_refs: List of detected temporal references
            - date_context: Current date context string for LLM
            - has_temporal: Boolean indicating if temporal refs found
    
    Example:
        >>> result = parse_datetime_query("What happened today with UPI?")
        >>> print(result['has_temporal'])
        True
        >>> print(result['augmented_query'])
        "What happened today with UPI? [today = 2026-03-06]"
    """
    parser = DateTimeParser()
    detected = parser.detect_temporal_references(query)
    
    return {
        'original_query': query,
        'augmented_query': parser.augment_query(query),
        'temporal_refs': detected,
        'date_context': parser.get_current_date_context(),
        'has_temporal': len(detected) > 0
    }


# For quick testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test queries
    test_queries = [
        "What happened today with UPI?",
        "What changed last week with NEFT?",
        "Any updates yesterday on RBI policies?",
        "What are the latest digital payment trends?",
        "Show me news from last 7 days",
        "What are RTGS charges?"  # No temporal reference
    ]
    
    print("=" * 80)
    print("DateTime Parser Test")
    print("=" * 80)
    
    parser = DateTimeParser()
    print(f"\n{parser.get_current_date_context()}\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = parse_datetime_query(query)
        
        if result['has_temporal']:
            print(f"  ✓ Temporal refs: {result['temporal_refs']}")
            print(f"  → Augmented: {result['augmented_query']}")
        else:
            print(f"  ✗ No temporal references detected")